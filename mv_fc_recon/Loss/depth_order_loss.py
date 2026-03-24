import torch
from typing import Optional 


def depth_order_loss(
    depth: torch.Tensor,             # [H, W] or [H, W, 1] or [1, H, W]
    gt_depth: torch.Tensor,          # same shape as depth
    mask: Optional[torch.Tensor] = None,   # [H, W] / [H, W, 1] / [1, H, W], 1 means valid
    gt_mask: Optional[torch.Tensor] = None,   # [H, W] / [H, W, 1] / [1, H, W], 1 means valid
    max_pixel_shift_ratio: float = 0.05,
    num_samples: Optional[int] = None,
    margin: float = 1e-4,
    reduction: str = "mean",
    log_space: bool = True,
    log_scale: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    深度排序损失：
    只要求 depth 与 gt_depth 在局部像素对上的前后顺序一致，
    不要求绝对尺度一致，因此比直接 depth regression 更适合 noisy prior depth。

    直观上：
        若 gt_depth(p) > gt_depth(q)，
        则希望 depth(p) > depth(q)

    这里假设“深度值越大表示越远”。
    如果你的定义相反（越大越近），把 diff 和 gt_diff 同时乘 -1 即可，或在外部先取负。

    Args:
        depth:      预测深度
        gt_depth:   参考深度 / prior depth
        mask:       有效区域掩码；若提供，则要求 p 和 q 都有效
        max_pixel_shift_ratio:
                    随机配对的最大位移比例，相对于 max(H, W)
        num_samples:
                    若为 None，则每个像素采一个配对点；
                    若给定，则从有效像素中随机采样 num_samples 个 anchor 点
        margin:     gt_depth 差值绝对值小于该阈值时，不监督（因为顺序不稳定）
        reduction:  "mean" / "sum" / "none"
        log_space:  是否对 loss 做 log(1 + s*x)，压缩大残差
        log_scale:  log_space 的尺度
        eps:        数值稳定项

    Returns:
        标量 loss；若 reduction="none"，返回逐样本 loss（shape [N]）
    """

    def _squeeze_depth(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            if x.shape[0] == 1:
                return x[0]
            if x.shape[-1] == 1:
                return x[..., 0]
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    depth = _squeeze_depth(depth)
    gt_depth = _squeeze_depth(gt_depth)

    if depth.shape != gt_depth.shape:
        raise ValueError(f"depth.shape {depth.shape} != gt_depth.shape {gt_depth.shape}")

    H, W = depth.shape
    device = depth.device

    if mask is None:
        valid_mask = torch.ones((H, W), dtype=torch.bool, device=device)
    else:
        mask = _squeeze_depth(mask)
        gt_mask = _squeeze_depth(gt_mask) 
        mask = mask.bool() & gt_mask.bool() 
        if mask.shape != depth.shape:
            raise ValueError(f"mask.shape {mask.shape} != depth.shape {depth.shape}")
        valid_mask = mask > 0.5

    # 有效像素坐标
    valid_coords = torch.nonzero(valid_mask, as_tuple=False)   # [Nv, 2]
    if valid_coords.shape[0] == 0:
        return depth.new_tensor(0.0)

    # 选择 anchor 像素
    if num_samples is None or num_samples >= valid_coords.shape[0]:
        anchor_coords = valid_coords
    else:
        rand_idx = torch.randint(
            low=0, high=valid_coords.shape[0], size=(num_samples,), device=device
        )
        anchor_coords = valid_coords[rand_idx]

    N = anchor_coords.shape[0]
    if N == 0:
        return depth.new_tensor(0.0)

    max_pixel_shift = max(round(max_pixel_shift_ratio * max(H, W)), 1)

    # 随机位移，避免 (0,0)
    shifts = torch.randint(
        low=-max_pixel_shift,
        high=max_pixel_shift + 1,
        size=(N, 2),
        device=device
    )
    zero_shift = (shifts[:, 0] == 0) & (shifts[:, 1] == 0)
    while zero_shift.any():
        shifts[zero_shift] = torch.randint(
            low=-max_pixel_shift,
            high=max_pixel_shift + 1,
            size=(zero_shift.sum(), 2),
            device=device
        )
        zero_shift = (shifts[:, 0] == 0) & (shifts[:, 1] == 0)

    paired_coords = anchor_coords + shifts
    paired_coords[:, 0].clamp_(0, H - 1)
    paired_coords[:, 1].clamp_(0, W - 1)

    # 要求 pair 两端都有效
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

    # gt 差值太小时，不监督，因为排序本来就不可靠
    strong_pair = gt_diff.abs() > margin
    if not strong_pair.any():
        return depth.new_tensor(0.0)

    pred_diff = pred_diff[strong_pair]
    gt_diff = gt_diff[strong_pair]

    # 只关心符号一致性，不关心尺度
    gt_sign = torch.sign(gt_diff)   # ±1

    # hinge ranking loss:
    # 希望 gt_sign * pred_diff > 0
    # 若符号相反，会产生正损失；若同号且足够大，损失为 0
    loss = torch.relu(-(gt_sign * pred_diff))

    # 可选 log 压缩，减少少数大错点的主导
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
