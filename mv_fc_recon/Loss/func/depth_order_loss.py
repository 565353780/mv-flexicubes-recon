import torch
from typing import Optional 


def fill_invalid_depth_with_boundary_depth_knn(
    depth: torch.Tensor,        # [H, W]
    pred_mask: torch.Tensor,    # [H, W], bool, True=pred valid
    fill_mask: torch.Tensor,    # [H, W], bool
    k: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert depth.ndim == 2
    assert pred_mask.ndim == 2
    assert fill_mask.ndim == 2
    assert depth.shape == pred_mask.shape == fill_mask.shape

    up = torch.zeros_like(pred_mask)
    down = torch.zeros_like(pred_mask)
    left = torch.zeros_like(pred_mask)
    right = torch.zeros_like(pred_mask)

    up[1:] = pred_mask[:-1]
    down[:-1] = pred_mask[1:]
    left[:, 1:] = pred_mask[:, :-1]
    right[:, :-1] = pred_mask[:, 1:]

    neighbor_diff = (pred_mask != up) | (pred_mask != down) | (pred_mask != left) | (pred_mask != right)
    boundary_mask = pred_mask & neighbor_diff

    by, bx = torch.where(boundary_mask)
    iy, ix = torch.where(fill_mask)

    if iy.numel() == 0:
        return depth

    if by.numel() == 0:
        by, bx = torch.where(pred_mask)
        if by.numel() == 0:
            return depth

    boundary_yx = torch.stack([by, bx], dim=1).float()   # [Nb,2]
    invalid_yx = torch.stack([iy, ix], dim=1).float()    # [Ni,2]

    dist = torch.cdist(invalid_yx, boundary_yx)          # [Ni,Nb]

    k = min(k, dist.shape[1])
    knn_dist, knn_idx = torch.topk(dist, k=k, dim=1, largest=False)   # [Ni,k]

    src_y = by[knn_idx]   # [Ni,k]
    src_x = bx[knn_idx]   # [Ni,k]

    src_depth = depth[src_y, src_x]   # [Ni,k]

    # 距离越近，权重越大
    weights = 1.0 / (knn_dist + eps)
    weights = weights / weights.sum(dim=1, keepdim=True)

    filled_values = (weights * src_depth).sum(dim=1)   # [Ni]

    filled = depth.clone()
    filled[iy, ix] = filled_values
    return filled




def depth_order_loss(
    _depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,      # pred valid mask
    gt_mask: Optional[torch.Tensor] = None,   # gt mask
    max_pixel_shift_ratio: float = 0.05,
    num_samples: Optional[int] = None,
    margin: float = 1e-4,
    reduction: str = "mean",
    log_space: bool = True,
    log_scale: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:

    def _squeeze_depth(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            if x.shape[0] == 1:
                return x[0]
            if x.shape[-1] == 1:
                return x[..., 0]
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    depth = _squeeze_depth(_depth)
    gt_depth = _squeeze_depth(gt_depth)

    if depth.shape != gt_depth.shape:
        raise ValueError(f"depth.shape {depth.shape} != gt_depth.shape {gt_depth.shape}")

    H, W = depth.shape
    device = depth.device

    pred_mask = _squeeze_depth(mask).bool() if mask is not None else torch.ones((H, W), dtype=torch.bool, device=device)

    if gt_mask is not None:
        gt_mask = _squeeze_depth(gt_mask).bool()
    else:
        gt_mask = pred_mask

    # 只填 GT有但pred没有 的地方
    fill_mask = gt_mask & (~pred_mask)

    # depth = fill_invalid_depth_with_boundary_depth(depth, pred_mask, fill_mask)
    depth = fill_invalid_depth_with_boundary_depth_knn(depth, pred_mask, fill_mask)
    

    # 监督区域只用：pred已有 + 刚填的缺失区
    # valid_mask = pred_mask | fill_mask
    valid_mask = fill_mask | (pred_mask & gt_mask)

    valid_coords = torch.nonzero(valid_mask, as_tuple=False)
    if valid_coords.shape[0] == 0:
        return depth.new_tensor(0.0)

    if num_samples is None or num_samples >= valid_coords.shape[0]:
        anchor_coords = valid_coords
    else:
        rand_idx = torch.randint(0, valid_coords.shape[0], (num_samples,), device=device)
        anchor_coords = valid_coords[rand_idx]

    N = anchor_coords.shape[0]
    if N == 0:
        return depth.new_tensor(0.0)

    max_pixel_shift = max(round(max_pixel_shift_ratio * max(H, W)), 1)

    shifts = torch.randint(-max_pixel_shift, max_pixel_shift + 1, (N, 2), device=device)
    zero_shift = (shifts[:, 0] == 0) & (shifts[:, 1] == 0)
    while zero_shift.any():
        shifts[zero_shift] = torch.randint(
            -max_pixel_shift, max_pixel_shift + 1, (zero_shift.sum(), 2), device=device
        )
        zero_shift = (shifts[:, 0] == 0) & (shifts[:, 1] == 0)

    paired_coords = anchor_coords + shifts
    paired_coords[:, 0].clamp_(0, H - 1)
    paired_coords[:, 1].clamp_(0, W - 1)

    pair_valid = valid_mask[anchor_coords[:, 0], anchor_coords[:, 1]] & \
                 valid_mask[paired_coords[:, 0], paired_coords[:, 1]]

    if not pair_valid.any():
        return depth.new_tensor(0.0)

    anchor_coords = anchor_coords[pair_valid]
    paired_coords = paired_coords[pair_valid]

    d1 = depth[anchor_coords[:, 0], anchor_coords[:, 1]]
    d2 = depth[paired_coords[:, 0], paired_coords[:, 1]]

    g1 = gt_depth[anchor_coords[:, 0], anchor_coords[:, 1]]
    g2 = gt_depth[paired_coords[:, 0], paired_coords[:, 1]]

    pred_diff = d1 - d2
    gt_diff = g1 - g2

    strong_pair = gt_diff.abs() > margin
    if not strong_pair.any():
        return depth.new_tensor(0.0)

    pred_diff = pred_diff[strong_pair]
    gt_diff = gt_diff[strong_pair]

    gt_sign = torch.sign(gt_diff)
    loss = torch.relu(-(gt_sign * pred_diff))

    if log_space:
        loss = torch.log1p(log_scale * loss)

    if reduction == "mean":
        return loss.mean() if loss.numel() > 0 else depth.new_tensor(0.0)
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    