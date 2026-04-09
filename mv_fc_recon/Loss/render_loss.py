import torch
from dataclasses import dataclass, field 

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mv_fc_recon.Loss.func.depth_order_loss import depth_order_loss
from mv_fc_recon.Loss.func.loss_utils import render_rgb_loss, BCELossFn
from mv_fc_recon.Loss.base.loss_base import EMABalancedLoss, BaseLoss 
from mv_fc_recon.Loss.base.context import StepContext, LossResult, BaseLossConfig


@dataclass
class RenderLossConfig(BaseLossConfig):
    lambda_rgb: float = 0.0
    lambda_depth: float = 1e1
    lambda_mask: float = 1e1
    log_render_limit: int = 9
    render_types: list = field(default_factory=lambda: ['mask', 'rgb', 'depth', 'normal'])


class RenderLoss(BaseLoss):
    name = "Render"

    def __init__(self, cfg: RenderLossConfig):
        super().__init__(cfg)
        self.lambda_rgb = cfg.lambda_rgb
        self.lambda_depth = cfg.lambda_depth
        self.lambda_mask = cfg.lambda_mask
        self.log_render_limit = getattr(cfg, "log_render_limit", 9)
        self.render_types = getattr(
            cfg,
            "render_types",
            ["mask", "rgb", "depth", "normal"],
        )

    def forward(self, ctx: StepContext) -> LossResult:
        device = ctx.vertices.device

        total_mask_loss = torch.tensor(0.0, device=device)
        total_depth_loss = torch.tensor(0.0, device=device)
        total_color_loss = torch.tensor(0.0, device=device)

        render_data_list = []
        render_idx_list = []

        num_cameras = len(ctx.camera_list)
        batch_indices = list(range(num_cameras))

        for idx in batch_indices:
            camera = ctx.camera_list[idx]
            target_data = ctx.targets[idx]

            target_mask_data = target_data["target_mask"]
            target_depth_data = target_data["target_depth"]
            target_color_data = target_data["target_color"]

            render_dict = NVDiffRastRenderer.render(
                mesh=ctx.mesh,
                camera=camera,
                render_types=self.render_types,
                vertices_tensor=ctx.vertices,
                enable_antialias=True,
            ) 

            mask_data = render_dict["mask"]
            mask_bool = mask_data.bool()
            normal_data = render_dict["rgb_normal_world"]
            depth_data = render_dict["depth"]

            # 保留 rgb_data，后续你继续改表观逻辑时直接改这里
            rgb_data = render_dict.get("rgb", None)

            if self.lambda_mask > 0:
                mask_render_loss = BCELossFn(mask_data.float(), target_mask_data.float())
                total_mask_loss = total_mask_loss + self.lambda_mask * mask_render_loss

            if self.lambda_depth > 0:
                total_depth_loss = total_depth_loss + self.lambda_depth * depth_order_loss(
                    depth_data,
                    target_depth_data,
                    mask_bool,
                    target_mask_data,
                )

            if self.lambda_rgb > 0:
                if rgb_data is None:
                    raise ValueError("lambda_rgb > 0 but render_dict does not contain 'rgb'")
                vertex_render_loss = render_rgb_loss(rgb_data, target_color_data)
                total_color_loss = total_color_loss + self.lambda_rgb * vertex_render_loss


            if len(render_data_list) < self.log_render_limit:
                if self.lambda_rgb > 0 and rgb_data is not None:
                    render_data_list.append([normal_data.detach().clone(), rgb_data.detach().clone()])
                else:
                    render_data_list.append([normal_data.detach().clone()])
                render_idx_list.append(idx)


        avg_render_loss = (
            total_color_loss + total_mask_loss + total_depth_loss
        ) / len(batch_indices)

        # 给后面的 EMA regularizer 用
        ctx.shared_state["avg_render_loss"] = avg_render_loss.detach()

        return LossResult(
            loss=self.loss_lambda * avg_render_loss,
            metrics={
                "Render": avg_render_loss.item(),
                "l_mask": (total_mask_loss / len(batch_indices)).item(),
                "l_depth": (total_depth_loss / len(batch_indices)).item(),
                "l_color": (total_color_loss / len(batch_indices)).item(),
            },
            artifacts={
                "render_data_list": render_data_list,
                "render_idx_list": render_idx_list,
            },
        )

