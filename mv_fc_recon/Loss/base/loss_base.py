from abc import ABC, abstractmethod
from typing import Any, List
import torch

from .context import StepContext, LossResult, EMABalancedLossConfig, BaseLossConfig


class BaseLoss(ABC):
    name = "base_loss"

    def __init__(self, cfg: BaseLossConfig):
        self.cfg = cfg
        self.loss_lambda = cfg.loss_lambda 

    def __call__(self, ctx: StepContext) -> LossResult:
        return self.forward(ctx)

    @abstractmethod
    def forward(self, ctx: StepContext) -> LossResult:
        pass



class EMABalancedLoss(BaseLoss, ABC):
    name = "ema_base_loss"

    def __init__(self, cfg: EMABalancedLossConfig):
        super().__init__(cfg)
        self.warmup_start = cfg.ema_warmup_start
        self.warmup_len = cfg.ema_warmup_len
        self.ema_alpha = cfg.ema_alpha
        self.target_ratio = cfg.target_ratio
        self.ema_lambda = None

    def compute_current_lambda(
        self,
        iteration: int,
        raw_loss: torch.Tensor,
        ref_loss: torch.Tensor,
    ) -> torch.Tensor:
        if iteration < self.warmup_start:
            zero = torch.zeros((), device=raw_loss.device, dtype=raw_loss.dtype)
            return zero

        raw_lambda = (
            self.target_ratio * ref_loss.detach() / (raw_loss.detach() + 1e-8)
        ).detach()

        if self.ema_lambda is None:
            self.ema_lambda = raw_lambda
        else:
            self.ema_lambda = (
                (1.0 - self.ema_alpha) * self.ema_lambda
                + self.ema_alpha * raw_lambda
            ).detach()

        lambda_cur = self.ema_lambda
        t = min((iteration - self.warmup_start) / max(self.warmup_len, 1), 1.0)
        lambda_cur = lambda_cur * t
        return lambda_cur

    def forward(self, ctx: StepContext) -> LossResult:
        raw_loss = self.compute_raw_loss(ctx)
        ref_loss = self.get_reference_loss(ctx).detach()

        lambda_cur = self.compute_current_lambda(
            iteration=ctx.iteration,
            raw_loss=raw_loss,
            ref_loss=ref_loss,
        )

        weighted_loss = self.loss_lambda * lambda_cur * raw_loss

        return LossResult(
            loss=weighted_loss,
            metrics={
                f"{self.name}_raw": raw_loss.item(),
                self.name: weighted_loss.item(),
                f"lambda_{self.name}": float(lambda_cur),
            }
        )

    @abstractmethod
    def compute_raw_loss(self, ctx: StepContext) -> torch.Tensor:
        pass

    @abstractmethod
    def get_reference_loss(self, ctx: StepContext) -> torch.Tensor:
        pass
