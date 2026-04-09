import torch 
from typing import Any, Dict 
from dataclasses import dataclass, field
from mv_fc_recon.Loss.func.sdf_reg_loss import sdf_hessian_energy_loss
from mv_fc_recon.Loss.func.thin_plate_energy import thin_plate_energy 
from mv_fc_recon.Loss.func.sdf_reg_loss import flexicube_sdf_reg_loss
from mv_fc_recon.Loss.base.loss_base import EMABalancedLoss, BaseLoss 
from mv_fc_recon.Loss.base.context import StepContext, LossResult, EMABalancedLossConfig, BaseLossConfig

@dataclass
class ThinPlateLossConfig(EMABalancedLossConfig):
    target_ratio: float = 3e-4


@dataclass
class HessianLossConfig(EMABalancedLossConfig):
    target_ratio: float = 2e-3


@dataclass
class FlexiCubesRegLossConfig(BaseLossConfig):
    loss_lambda: float = 0.2



class ThinPlateLoss(EMABalancedLoss):
    name = "TP_loss"
    def __init__(self, cfg: ThinPlateLossConfig):
        super().__init__(cfg) 

    def compute_raw_loss(self, ctx: StepContext) -> torch.Tensor:
        gctx = ctx.geometry()
        faces_tensor = torch.from_numpy(gctx.mesh.faces).long().to(ctx.device)

        return thin_plate_energy(gctx.vertices, faces_tensor, factor=1)

    def get_reference_loss(self, ctx: StepContext) -> torch.Tensor:
        return ctx.shared_state["avg_render_loss"] 
    


class HessianLoss(EMABalancedLoss):
    name = "HES_loss"
    def __init__(self, cfg: HessianLossConfig):
        super().__init__(cfg) 

    def compute_raw_loss(self, ctx: StepContext) -> torch.Tensor:
        gctx = ctx.geometry()
        return sdf_hessian_energy_loss(
            gctx.sdf,
            gctx.fc_params["grid_edges"],
            gctx.fc_params["x_nx3"],
        )

    def get_reference_loss(self, ctx: StepContext) -> torch.Tensor:
        return ctx.shared_state["avg_render_loss"] 
    


class FlexiCubesRegLoss(BaseLoss):
    name = "Reg_loss"

    def __init__(self, cfg: FlexiCubesRegLossConfig):
        super().__init__(cfg)

    def forward(self, ctx: StepContext) -> LossResult:
        device = ctx.vertices.device
        zero = torch.tensor(0.0, device=device)

        if self.loss_lambda <= 0:
            return LossResult(
                loss=zero,
                metrics={self.name: 0.0},
            )

        gctx = ctx.geometry()

        t_iter = ctx.iteration / max(ctx.num_iterations, 1)
        sdf_weight = self.loss_lambda - (self.loss_lambda - self.loss_lambda / 20.0) * min(1.0, 4.0 * t_iter)

        reg_loss = flexicube_sdf_reg_loss(
            gctx.sdf,
            gctx.fc_params["grid_edges"]
        ).mean() * sdf_weight
        reg_loss = reg_loss + gctx.L_dev.mean() * 0.5 

        weight = gctx.fc_params["weight"]
        reg_loss = reg_loss + (weight[:, :20]).abs().mean() * 0.1 

        return LossResult(
            loss=reg_loss,
            metrics={
                self.name: reg_loss.item(),
            },
        )

