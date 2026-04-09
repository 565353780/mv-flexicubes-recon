from dataclasses import dataclass, field

from mv_fc_recon.Loss.base.composite_loss import CompositeLoss
from .render_loss import RenderLoss, RenderLossConfig
from .regularize_loss import (
    ThinPlateLoss, 
    HessianLoss, 
    FlexiCubesRegLoss, 
    ThinPlateLossConfig, 
    HessianLossConfig, 
    FlexiCubesRegLossConfig
)



@dataclass
class LossSystemConfig:
    render: RenderLossConfig = field(default_factory=RenderLossConfig)
    thin_plate: ThinPlateLossConfig = field(default_factory=ThinPlateLossConfig)
    hessian: HessianLossConfig = field(default_factory=HessianLossConfig)
    reg: FlexiCubesRegLossConfig = field(default_factory=FlexiCubesRegLossConfig)


def build_loss_system(cfg: LossSystemConfig) -> CompositeLoss:
    return CompositeLoss([
        RenderLoss(cfg.render),
        ThinPlateLoss(cfg.thin_plate),
        HessianLoss(cfg.hessian),
        FlexiCubesRegLoss(cfg.reg),
    ])

