import os
import torch
import trimesh
from tqdm import tqdm
from typing import Tuple, Union, List, Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from camera_control.Module.camera import Camera


from flexi_cubes.Module.fc_convertor import FCConvertor
from flexi_cubes.Module.sh_utils import RGB2SH, SH2RGB, eval_sh 

from mv_fc_recon.Method.exportSHMesh import bake_vertex_colors_from_sh

# from mv_fc_recon.Loss.all_loss import loss_system 
from mv_fc_recon.Loss.base.context import StepContext, LossResult

from mv_fc_recon.Loss.base.composite_loss import CompositeLoss
from mv_fc_recon.Loss.loss_factory import LossSystemConfig, build_loss_system 

from dataclasses import dataclass, field



@dataclass 
class TrainerConfig:
    camera_list: List[Camera]
    initMesh: trimesh.Trimesh | None = None

    lossSystemConfig: LossSystemConfig = field(default_factory=LossSystemConfig)
    device: str = "cuda:0"
    num_iterations: int = 120
    lr: float = 5e-4
    resolution: int = 128
    bg_color: list = field(default_factory=lambda: [255, 255, 255])
    log_image_num: int = 20
    log_interval: int = 10
    log_dir: str = "./output/"



class Trainer(object):
    def __init__(self, cfg: TrainerConfig) -> None:
        self.loss_system: CompositeLoss = build_loss_system(cfg.lossSystemConfig) 
        self.lossConfig = cfg.lossSystemConfig 
        self.device: str = cfg.device 
        self.num_iterations: int = cfg.num_iterations
        self.lr: float = cfg.lr
        self.camera_list: List[Camera] = cfg.camera_list
        self.initMesh: trimesh.Trimesh = cfg.initMesh
        self.resolution = cfg.resolution
        self.bg_color = cfg.bg_color
        self.log_image_num = cfg.log_image_num
        self.log_interval = cfg.log_interval
        self.log_dir = cfg.log_dir

        self.sh_resolution = 128 

        #------准备训练数据和上下文------
        self.target_data_list = [] 
        self.fc_params = None 
        self.log_writer = None
        self.initDatasetContext() 
        if self.fc_params is None:
            raise ValueError("Failed to initialize FC parameters.")
        self.optimizer = Trainer.createOptimizer(self.fc_params, lr=self.lr) 


    def initDatasetContext(self): 
        initColor = True if self.lossConfig.render.lambda_rgb > 0 else False 
        self.fc_params = FCConvertor.createFC(
            self.initMesh, self.resolution, self.sh_resolution, self.device, 
            initColor=initColor) 
        if self.fc_params is None:
            return None

        for camera in self.camera_list: 
            camera.to(device=self.device)
            target_mask = camera.mask.float() 
            target_depth = camera.toDepth(use_mask=True)
            target_color = camera.toImageVis(use_mask=True)
            target_color_vis = camera.toImageVis(use_mask=True)

            self.target_data_list.append({
                "target_mask": target_mask, 
                "target_depth": target_depth,
                "target_color": target_color, 
                "target_color_vis": target_color_vis, 
            }) 

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            # tensorboard writer 
            self.log_writer = SummaryWriter(log_dir=self.log_dir)
            for i, target_data in enumerate(self.target_data_list):
                if i >= self.log_image_num:
                    break
                self.log_writer.add_image(f'GT/Camera_{i}', (target_data["target_color_vis"]).transpose(2, 0, 1), global_step=0)
                if self.lossConfig.render.lambda_rgb > 0: 
                    self.log_writer.add_image(f'GT_color/Camera_{i}', target_data["target_color"].clone().permute(2, 0, 1), global_step=0)

            # 记录初始 mesh
            with torch.no_grad():
                curr_mesh, _, _, _, verts_sh_coeff = FCConvertor.extractMesh(self.fc_params, training=True) 
                if curr_mesh is not None and len(curr_mesh.vertices) > 0 and len(curr_mesh.faces) > 0:
                    try:
                        curr_mesh.export(self.log_dir + 'start_fc_mesh.ply') 
                    except Exception as e:
                        print(f'[WARNING] Failed to export start mesh: {e}')

    @staticmethod
    def createOptimizer(
        fc_params: Dict,
        lr: float = 0.01,
        lr_sdf: Optional[float] = None,
        lr_deform: Optional[float] = None,
        lr_weight: Optional[float] = None,
        lr_sh: Optional[float] = None
    ) -> torch.optim.Adam:
        """
        为FlexiCubes参数创建优化器

        Args:
            fc_params: createFC返回的参数字典
            lr: 默认学习率
            lr_sdf: SDF学习率（可选）
            lr_deform: 变形学习率（可选）
            lr_weight: 权重学习率（可选）

        Returns:
            optimizer: Adam优化器
        """
        if lr_sdf is None:
            lr_sdf = lr
        if lr_deform is None:
            lr_deform = lr
        if lr_weight is None:
            lr_weight = lr
        if lr_sh is None: 
            lr_sh = lr 

        if fc_params['sh_coeff'] is not None: 
            print("[info] optimization with sh_coeff. ")
            param_groups = [
                dict(params=[fc_params['sdf']], lr=lr_sdf),
                dict(params=[fc_params['deform']], lr=lr_deform),
                dict(params=[fc_params['weight']], lr=lr_weight),
                dict(params=[fc_params['sh_coeff']], lr=lr_sh)
            ]
        else: 
            print("[info] optimize geometry only. ")
            param_groups = [
                dict(params=[fc_params['sdf']], lr=lr_sdf),
                dict(params=[fc_params['deform']], lr=lr_deform),
                dict(params=[fc_params['weight']], lr=lr_weight)
            ]

        optimizer = torch.optim.Adam(param_groups)
        return optimizer


    def train_one_step(self, iter, current_mesh, vertices_tensor, L_dev, sdf, verts_sh_coeff) -> Tuple[bool, Optional[LossResult]]: 
        # 创建损失上下文 
        step_ctx = StepContext(
            iteration=iter,
            num_iterations=self.num_iterations,
            device=self.device,
            mesh=current_mesh,
            vertices=vertices_tensor,
            sdf=sdf,
            L_dev=L_dev,
            fc_params=self.fc_params,
            camera_list=self.camera_list,
            targets=self.target_data_list
        )
        
        loss_result = self.loss_system(step_ctx) 
        return True, loss_result 


    def train_loop(self): 
        # 训练循环
        pbar = tqdm(range(self.num_iterations), desc='FlexiCubes Optimization') 
        for iteration in pbar:
            self.optimizer.zero_grad() 

            # 从 FlexiCubes 参数提取 mesh
            current_mesh, vertices_tensor, L_dev, sdf, verts_sh_coeff = FCConvertor.extractMesh(self.fc_params, training=True) 

            if current_mesh is None or len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                print(f'[WARNING] Invalid mesh at iteration {iter}, skipping...')
                continue 
            
            isValid, loss_result = self.train_one_step(
                iteration, 
                current_mesh, vertices_tensor, L_dev, sdf, verts_sh_coeff
            )
            total_loss = loss_result.loss
            loss_dict = loss_result.metrics
            artifacts = loss_result.artifacts
            
            # 检查损失是否包含 NaN 或 Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f'[WARNING] Invalid loss (NaN/Inf) at iteration {iteration}, skipping...')
                continue
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [self.fc_params['sdf'], self.fc_params['deform'], self.fc_params['weight']],
                max_norm=1.0
            )

            # 检查梯度是否包含 NaN 或 Inf
            has_nan_grad = False
            has_inf_grad = False
            for param in [self.fc_params['sdf'], self.fc_params['deform'], self.fc_params['weight']]:
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
                    if has_nan_grad and has_inf_grad:
                        break

            if has_nan_grad:
                print(f'[WARNING] NaN gradients at iteration {iteration}, skipping update...')

            if has_inf_grad:
                print(f'[WARNING] Inf gradients at iteration {iteration}, skipping update...')

            if has_nan_grad or has_inf_grad:
                exit()

            # 更新参数
            self.optimizer.step()

            # 裁剪 SDF 值
            with torch.no_grad():
                self.fc_params['sdf'].data = torch.clamp(self.fc_params['sdf'].data, -10.0, 10.0)

            if self.log_writer is not None:
                self.log_writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                # 记录所有实际计算的 loss
                for loss_name, loss_value in loss_dict.items():
                    self.log_writer.add_scalar(f'Loss/{loss_name}', loss_value, iteration)

            # 更新进度条和日志
            if iteration % self.log_interval == 0 or iteration == self.num_iterations - 1:
                if self.log_writer is not None:
                    # 记录渲染图像
                    for i, render_data_group in enumerate(artifacts['render_data_list']):
                        for j, render_data in enumerate(render_data_group): 
                            if render_data.dim() == 3 and render_data.shape[-1] == 3:
                                render_data = render_data.permute(2, 0, 1)
                            self.log_writer.add_image(f'Render/Camera_{i}_{j}', render_data, global_step=iteration)

            postfix_dict = {'loss': f'{total_loss.item():.4f}'}
            pbar.set_postfix(postfix_dict)



    def fit(self) -> Optional[trimesh.Trimesh]:
        self.train_loop() 

        # 关闭 TensorBoard writer
        if self.log_writer is not None:
            self.log_writer.close()

        # 提取最终 mesh
        final_mesh, final_vertices, _, _, verts_sh_coeff = FCConvertor.extractMesh(self.fc_params, training=True)
        if verts_sh_coeff is not None: 
            final_mesh = bake_vertex_colors_from_sh(
                renderer=NVDiffRastRenderer(),
                camera_list=self.camera_list,
                final_mesh=final_mesh,
                final_vertices=final_vertices,
                verts_sh_coeff=verts_sh_coeff,
                sh_deg=self.fc_params["sh_deg"],
            )
        return final_mesh

