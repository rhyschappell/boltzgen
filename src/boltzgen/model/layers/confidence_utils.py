import torch
from torch import nn
from boltzgen.data import const


def compute_collinear_mask(v1, v2):
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity,
    resolved_mask=None,
    inference=False,
):
    with torch.amp.autocast("cuda", enabled=False):
        asym_id_token = feats["asym_id"]
        asym_id_atom = torch.bmm(
            feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()
        ).squeeze(-1)

    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    frames_idx_pred = (
        frames_idx_true.clone()
        .repeat_interleave(multiplicity, 0)
        .reshape(B // multiplicity, multiplicity, -1, 3)
    )

    # Iterate through the batch and modify the frames for nonpolymers
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if (
                feats["mol_type"][i, token_idx] != const.chain_type_ids["NONPOLYMER"]
                or num_atoms < 3
            ):
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5
            if inference:
                resolved_pair = 1 - (
                    feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                    * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            else:
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                resolved_pair = 1 - (
                    resolved_mask[i][mask_chain_atom.bool()][None, :]
                    * resolved_mask[i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            try:
                frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            except Exception as e:
                print(f"Failed to process {feats['pdb_id']} due to {e}")
            token_idx += num_tokens
            atom_idx += num_atoms

    frames_expanded = pred_atom_coords[
        torch.arange(0, B // multiplicity, 1)[:, None, None, None].to(
            frames_idx_pred.device
        ),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(
            frames_idx_pred.device
        ),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Compute masks for collinearity / overlap
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B // multiplicity, multiplicity, -1)
    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]


def compute_aggregated_metric(logits, end=1.0):
    # Compute aggregated metric from logits
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def tm_function(d, Nres):
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def compute_ptms(logits, x_preds, feats, multiplicity):
    # It needs to take as input the mask of the frames as they are not used to compute the PTM
    _, mask_collinear_pred = compute_frame_pred(
        x_preds, feats["frames_idx"], feats, multiplicity, inference=True
    )
    # mask overlapping, collinear tokens and ions (invalid frames)
    mask_pad = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    mask_pad_bool = mask_pad.bool()
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    pair_mask_iptm = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
    )
    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    end = 32.0
    pae_value = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    ).unsqueeze(0)
    N_res = mask_pad.sum(dim=-1, keepdim=True)
    tm_value = tm_function(pae_value, N_res).unsqueeze(1).unsqueeze(2)
    probs = nn.functional.softmax(logits, dim=-1)
    tm_expected_value = torch.sum(
        probs * tm_value,
        dim=-1,
    )  # shape (B, N, N)
    pae_expected_value = torch.sum(
        probs * pae_value.view(1, 1, 1, -1),
        dim=-1,
    )
    ptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_ptm, dim=-1)
        / (torch.sum(pair_mask_ptm, dim=-1) + 1e-5),
        dim=1,
    ).values
    iptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_iptm, dim=-1)
        / (torch.sum(pair_mask_iptm, dim=-1) + 1e-5),
        dim=1,
    ).values

    # compute ligand and protein iPTM
    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()

    ligand_iptm_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            (is_ligand_token[:, :, None] * is_protein_token[:, None, :])
            + (is_protein_token[:, :, None] * is_ligand_token[:, None, :])
        )
    )
    protein_iptm_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (is_protein_token[:, :, None] * is_protein_token[:, None, :])
    )
    ligand_iptm = torch.max(
        torch.sum(tm_expected_value * ligand_iptm_mask, dim=-1)
        / (torch.sum(ligand_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values
    protein_iptm = torch.max(
        torch.sum(tm_expected_value * protein_iptm_mask, dim=-1)
        / (torch.sum(protein_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    # compute design and target iPTM and PTM
    if feats["design_mask"].sum()>0:
        is_chain_design_token = feats["chain_design_mask"].float()
        is_target_token = 1 - feats["chain_design_mask"].float()
    else:
        is_chain_design_token = feats["design_mask"].float()
        is_target_token = 1 - feats["design_mask"].float()
    design_iptm_mask = (
        maski[:, :, None]
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            is_chain_design_token[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_chain_design_token[:, None, :]
        )
    )
    design_iptm = torch.max(
        torch.sum(tm_expected_value * design_iptm_mask, dim=-1)
        / (torch.sum(design_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    is_design_token = feats["design_mask"].float()
    is_target_token = 1 - is_chain_design_token

    design_to_target_iptm_mask = (
        maski[:, :, None]
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            is_design_token[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_design_token[:, None, :]
        )
    )
    design_to_target_iptm = torch.max(
        torch.sum(tm_expected_value * design_to_target_iptm_mask, dim=-1)
        / (torch.sum(design_to_target_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    design_ptm_mask = (
        maski[:, :, None]
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (is_chain_design_token[:, :, None] * is_chain_design_token[:, None, :])
    )
    target_ptm_mask = (
        maski[:, :, None]
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (is_target_token[:, :, None] * is_target_token[:, None, :])
    )

    design_ptm = torch.max(
        torch.sum(tm_expected_value * design_ptm_mask, dim=-1)
        / (torch.sum(design_ptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values
    target_ptm = torch.max(
        torch.sum(tm_expected_value * target_ptm_mask, dim=-1)
        / (torch.sum(target_ptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    # Compute pair chain ipTM
    chain_pair_iptm = {}
    asym_ids_list = torch.unique(asym_id).tolist()
    for idx1 in asym_ids_list:
        chain_iptm = {}
        for idx2 in asym_ids_list:
            mask_pair_chain = (
                maski[:, :, None]
                * (asym_id[:, None, :] == idx1)
                * (asym_id[:, :, None] == idx2)
                * mask_pad[:, None, :]
                * mask_pad[:, :, None]
            )

            chain_iptm[idx2] = torch.max(
                torch.sum(tm_expected_value * mask_pair_chain, dim=-1)
                / (torch.sum(mask_pair_chain, dim=-1) + 1e-5),
                dim=1,
            ).values
        chain_pair_iptm[idx1] = chain_iptm

    # Compute mask of part of design that has an atom within X Angstrom of any target atom
    x_pred_half = x_preds.bfloat16()
    interact_mask = torch.zeros_like(feats["design_mask"])
    if (
        is_chain_design_token.sum() > 0
        and (1 - is_chain_design_token.int()).sum() > 0
    ):
        atom_chain_design_mask = (torch.bmm(    feats["atom_to_token"].bfloat16(),    feats["chain_design_mask"].bfloat16().unsqueeze(-1),).squeeze(-1).bool())
        x_target = x_pred_half[~atom_chain_design_mask]
        x_design = x_pred_half[atom_chain_design_mask]
        dists = torch.cdist(x_target, x_design)
        atom_interact_mask = torch.zeros_like(atom_chain_design_mask)
        atom_interact_mask[atom_chain_design_mask] = (dists.min(0)[0] < 8).unsqueeze(0)
        index = torch.argmax(feats["atom_to_token"].int(), dim=-1)
        interact_mask.scatter_(dim=1, index=index, src=atom_interact_mask)

    # Compute interface interaction pTM values: iipTM
    # Any ptm of the design that is interacting with the target to anything in the target
    is_interacting = interact_mask.float()
    is_target_token = 1 - is_chain_design_token.float()
    design_iiptm_mask = (
        maski[:, :, None]
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            is_interacting[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_interacting[:, None, :]
        )
    )
    design_iiptm = torch.max(
        torch.sum(tm_expected_value * design_iiptm_mask, dim=-1)
        / (torch.sum(design_iiptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    # Compute ipSAE between design and target groups
    chain_design_mask_ipsae = (
        feats["chain_design_mask"]
        if feats["design_mask"].sum() > 0
        else feats["design_mask"]
    )
    if chain_design_mask_ipsae.shape[0] != mask_pad_bool.shape[0]:
        chain_design_mask_ipsae = chain_design_mask_ipsae.repeat_interleave(
            multiplicity, 0
        )
    design_mask_ipsae = chain_design_mask_ipsae.bool() & mask_pad_bool
    target_mask_ipsae = (~design_mask_ipsae) & mask_pad_bool
    pae_cutoff = 10.0
    if "ipsae_pae_cutoff" in feats:
        cutoff_value = feats["ipsae_pae_cutoff"]
        if isinstance(cutoff_value, torch.Tensor):
            pae_cutoff = float(cutoff_value.item())
        else:
            pae_cutoff = float(cutoff_value)
    design_to_target_ipsae = _compute_ipsae_from_masks(
        pae_expected_value, design_mask_ipsae, target_mask_ipsae, pae_cutoff
    )

    return (
        ptm,
        iptm,
        ligand_iptm,
        protein_iptm,
        chain_pair_iptm,
        design_to_target_iptm,
        design_iptm,
        design_iiptm,
        target_ptm,
        design_ptm,
        design_to_target_ipsae,
    )


def _compute_ipsae_from_masks(pae, source_mask, target_mask, pae_cutoff):
    scores = torch.zeros(pae.shape[0], device=pae.device, dtype=pae.dtype)
    for idx in range(pae.shape[0]):
        src_idx = torch.nonzero(source_mask[idx], as_tuple=False).squeeze(-1)
        tgt_idx = torch.nonzero(target_mask[idx], as_tuple=False).squeeze(-1)
        if src_idx.numel() == 0 or tgt_idx.numel() == 0:
            continue
        score_ab = _directional_ipsae(pae[idx], src_idx, tgt_idx, pae_cutoff)
        score_ba = _directional_ipsae(pae[idx], tgt_idx, src_idx, pae_cutoff)
        scores[idx] = min(score_ab, score_ba)
    return scores


def _directional_ipsae(pae, source_indices, partner_indices, pae_cutoff):
    best = torch.tensor(0.0, device=pae.device, dtype=pae.dtype)
    for src in source_indices.tolist():
        interface_pae = pae[src, partner_indices]
        mask = interface_pae < pae_cutoff
        if not torch.any(mask):
            continue
        partner_count = int(mask.sum().item())
        d0 = _calc_d0(partner_count)
        ptm_values = _ptm(interface_pae[mask], d0)
        best = torch.maximum(best, ptm_values.mean())
    return float(best.item())


def _ptm(pae_values, d0):
    return 1.0 / (1.0 + (pae_values / d0) ** 2)


def _calc_d0(length, pair_type="protein"):
    length = max(int(length), 27)
    min_value = 1.0 if pair_type != "nucleic_acid" else 2.0
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.num_gaussians = num_gaussians
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).reshape(
            *shape, self.num_gaussians
        )
