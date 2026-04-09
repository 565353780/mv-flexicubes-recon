from typing import List
from .context import StepContext, LossResult
from .loss_base import BaseLoss
import torch 

class CompositeLoss:
    def __init__(self, losses: List[BaseLoss]):
        self.losses = losses

    def __call__(self, ctx: StepContext) -> LossResult:
        total_loss = torch.tensor(0.0, device=ctx.vertices.device)
        metrics = {}
        artifacts = {}

        for loss_fn in self.losses:
            print(loss_fn.name) 
            result = loss_fn(ctx)
            total_loss = total_loss + result.loss
            metrics.update(result.metrics)

            if result.artifacts:
                artifacts.update(result.artifacts)

        return LossResult(
            loss=total_loss,
            metrics=metrics,
            artifacts=artifacts,
        ) 
