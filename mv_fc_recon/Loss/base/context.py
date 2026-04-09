import torch
from dataclasses import dataclass, field
from typing import Any, Dict 
import trimesh 


#-------config----------
@dataclass
class BaseLossConfig:
    loss_lambda: float = 1.0


@dataclass
class EMABalancedLossConfig(BaseLossConfig):
    target_ratio: float = 1e-3
    ema_warmup_start: int = 100
    ema_warmup_len: int = 40
    ema_alpha: float = 0.02


#-------context----------
@dataclass
class RenderContext:
    camera_batch: list
    targets: list

@dataclass
class GeometryContext:
    mesh: trimesh.Trimesh
    vertices: torch.Tensor
    fc_params: dict
    L_dev: torch.Tensor
    sdf: torch.Tensor

@dataclass
class TrainStateContext:
    iteration: int
    num_iterations: int
    device: str
    shared_state: dict


@dataclass
class LossResult:
    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepContext:
    iteration: int
    num_iterations: int
    device: str
    
    mesh: trimesh.Trimesh
    vertices: torch.Tensor
    fc_params: dict
    L_dev: torch.Tensor
    sdf: torch.Tensor
    
    camera_list: list
    targets: list 

    shared_state: Dict[str, Any] = field(default_factory=dict)

    def geometry(self):
        return GeometryContext(
            mesh=self.mesh,
            vertices=self.vertices,
            L_dev=self.L_dev,
            fc_params=self.fc_params,
            sdf=self.sdf
        )

    def render(self):
        return RenderContext(
            camera_batch=self.camera_batch,
            targets=self.targets,
        )

    def train_state(self):
        return TrainStateContext(
            iteration=self.iteration,
            num_iterations=self.num_iterations,
            device=self.device,
            shared_state=self.shared_state or {},
        )

