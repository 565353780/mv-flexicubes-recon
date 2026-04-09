import os
import torch
import trimesh
from tqdm import tqdm
from typing import Union, List, Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer
# from camera_control.Module.nvdiffrast_rendererV2 import NVDiffRastRenderer
from flexi_cubes.Module.fc_convertor import FCConvertor
from flexi_cubes.Module.sh_utils import RGB2SH, SH2RGB, eval_sh 

from mv_fc_recon.Loss.func.sdf_reg_loss import flexicube_sdf_reg_loss, sdf_hessian_energy_loss
from mv_fc_recon.Loss.func.thin_plate_energy import thin_plate_energy 
from mv_fc_recon.Loss.func.depth_order_loss import depth_order_loss 
from mv_fc_recon.Loss.func.loss_utils import l1_loss, render_rgb_loss, BCELossFn
from mv_fc_recon.Method.exportSHMesh import bake_vertex_colors_from_sh



class Trainer(object):
    def __init__(self) -> None:
        return

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

    @staticmethod
    def fitImagesWithSDFLoss(
        camera_list: List[Camera],
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 128,
        device: str = 'cuda:0',
        bg_color: list = [255, 255, 255],
        num_iterations: int = 120,
        lr: float = 5e-4,
        # 优化参数
        lambda_rgb: float = 0,  ## 1
        lambda_depth: float = 1e1, 
        lambda_mask: float = 1e1, 
        target_ratio_thin = 3e-4, 
        target_ratio_hes = 2e-3, 
        lambda_reg: float = 0.2 , # FlexiCubes equation(8)&(9) 默认正则化
        # 其他参数
        log_image_num: int = 20, 
        log_interval: int = 10,
        log_dir: str = './output/',
    ) -> trimesh.Trimesh:
        """通过多视角图像拟合 FlexiCubes 参数

        Returns:
            拟合后的 mesh
        """
        # 创建 FlexiCubes 参数

        initColor = True if lambda_rgb > 0 else False 
        fc_params = FCConvertor.createFC(mesh, resolution, device, initColor=initColor) 
        if fc_params is None:
            return None

        # 将相机移动到指定设备并预处理目标图像
        target_data_list = []
        for camera in camera_list: 
            camera.to(device=device)
            target_normal = camera.normal_world
            target_mask = camera.mask.float() 
            target_depth = camera.depth 
            target_color = camera.image 

            target_normal_vis = camera.toNormalWorldVisCV()

            target_data_list.append({
                "target_normal": target_normal, 
                "target_normal_vis": target_normal_vis,
                "target_mask": target_mask, 
                "target_depth": target_depth,
                "target_color": target_color, 
            }) 

        # 创建优化器
        optimizer = Trainer.createOptimizer(fc_params, lr=lr)

        # 获取 grid_edges 用于 SDF 正则化
        grid_edges = fc_params['grid_edges']

        # 创建 TensorBoard writer
        writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            for i, target_data in enumerate(target_data_list):
                if i >= log_image_num:
                    break
                # writer.add_image(f'GT/Camera_{i}', (target_data["target_normal"]).clone().permute(2, 0, 1), global_step=0)
                writer.add_image(f'GT/Camera_{i}', (target_data["target_normal_vis"]).transpose(2, 0, 1), global_step=0)
                if lambda_rgb > 0: 
                    writer.add_image(f'GT_color/Camera_{i}', target_data["target_color"].clone().permute(2, 0, 1), global_step=0)
        # print("pause. ")
        # input() 
        if log_dir:
            with torch.no_grad():
                curr_mesh, _, _, _, _ = FCConvertor.extractMesh(fc_params, training=True) 
                if curr_mesh is not None and len(curr_mesh.vertices) > 0 and len(curr_mesh.faces) > 0:
                    try:
                        curr_mesh.export(log_dir + 'start_fc_mesh.ply') 
                    except Exception as e:
                        print(f'[WARNING] Failed to export start mesh: {e}')


        # 训练循环
        pbar = tqdm(range(num_iterations), desc='FlexiCubes Optimization') 
        # renderer = NVDiffRastRenderer() 
        for iteration in pbar:
            optimizer.zero_grad()

            # 从 FlexiCubes 参数提取 mesh
            current_mesh, vertices_tensor, L_dev, sdf, verts_sh_coeff = FCConvertor.extractMesh(fc_params, training=True) 

            # 检查网格有效性
            if current_mesh is None or len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                print(f'[WARNING] Invalid mesh at iteration {iteration}, skipping...')
                continue

            # 获取 faces tensor 用于网格平滑
            faces_tensor = torch.from_numpy(current_mesh.faces).long().to(device)

            # ========== 计算实际使用的 Loss ==========
            # 计算总损失（只包含权重 > 0 的项）
            total_loss = torch.tensor(0.0, device=device)
            loss_dict = {}  # 用于 TensorBoard 记录

            # ========== 渲染损失 ==========
            total_mask_loss = torch.tensor(0.0, device=device)
            total_depth_loss = torch.tensor(0.0, device=device) 
            total_color_loss = torch.tensor(0.0, device=device) 
            render_data_list = []
            render_idx_list = []

            avg_render_loss = torch.tensor(0.0, device=device)
            if lambda_mask > 0 or lambda_depth > 0: 
                num_cameras = len(camera_list)
                batch_indices = list(range(num_cameras))

                for idx in batch_indices:
                    camera = camera_list[idx]
                    target_mask_data = target_data_list[idx]["target_mask"] 
                    target_depth_data = target_data_list[idx]["target_depth"] 
                    target_color_data = target_data_list[idx]["target_color"] 

                    render_dict = NVDiffRastRenderer.render(
                        mesh=current_mesh,
                        camera=camera,
                        render_types=['mask', 'rgb', 'depth', 'normal'],
                        # bg_color=bg_color,
                        vertices_tensor=vertices_tensor,
                        enable_antialias=True,
                    )
                    mask_data = render_dict['mask'] 
                    mask_bool = render_dict['mask'].bool() 
                    normal_data = render_dict['rgb_normal_world'] 
                    depth_data = render_dict['depth'] 
                    # rgb_data = render_dict['rgb'] 

                    
                    ##----------------------render losses-----------------------
                    ## l1: l_mask 
                    mask_render_loss = BCELossFn(mask_data.float(), target_mask_data.float()) 
                    total_mask_loss = total_mask_loss + lambda_mask * mask_render_loss 
                    ## l2: l_depth order
                    total_depth_loss = total_depth_loss + \
                        lambda_depth * depth_order_loss(depth_data, target_depth_data, mask_bool, target_mask_data) 
                    
                    if lambda_rgb > 0: 
                        # vertex_render_loss = ((render_data - target_color_data).abs()).mean()
                        vertex_render_loss = render_rgb_loss(rgb_data, target_color_data) ##gs l_rgb
                        total_color_loss = total_color_loss + lambda_rgb * vertex_render_loss
                     
                    ## log iamges
                    if len(render_data_list) < 9:
                        if lambda_rgb > 0: 
                            render_data_list.append([normal_data.clone(), rgb_data.clone()])
                        else: 
                            render_data_list.append([normal_data.clone() ])
                        render_idx_list.append(idx)
                   
                avg_render_loss = (total_color_loss + total_mask_loss + total_depth_loss ) / len(batch_indices)
                total_loss = total_loss + avg_render_loss
                loss_dict['Render'] = avg_render_loss.item()
                loss_dict['l_mask'] = (total_mask_loss / len(batch_indices)).item()
                loss_dict['l_depth'] = (total_depth_loss / len(batch_indices)).item() 
                loss_dict['l_color'] = (total_color_loss / len(batch_indices)).item()


            # ---------------------------------------------
            # Geometry regularizers: thin-plate + Hessian
            # 使用 warmup + clamp + EMA 动态平衡到 render loss
            # ---------------------------------------------
            if (target_ratio_thin > 0 or target_ratio_hes > 0):
            # if False: 
                warmup_start = 100 
                warmup_len = 40 

                # 统一用 avg_render_loss 作为参考主项
                render_ref = avg_render_loss.detach()

                # ----------------------------
                # 1) thin-plate regularizer
                # ----------------------------
                if target_ratio_thin > 0:
                # if False: 
                    thinplate_loss_raw = thin_plate_energy(
                        vertices_tensor, faces_tensor, factor=1
                    )

                    if iteration < warmup_start:
                        lambda_thin_plate_energy_cur = 0.0
                    else:
                        raw_lambda_thin = target_ratio_thin * render_ref / (thinplate_loss_raw.detach() + 1e-8)
                        # raw_lambda_thin = torch.clamp(raw_lambda_thin, min=0.0, max=1e-4).item()
                        alpha_thin = 0.02

                        if not hasattr(Trainer, "_lambda_thin_plate_energy_ema"):
                            Trainer._lambda_thin_plate_energy_ema = raw_lambda_thin

                        alpha_thin = 0.02
                        Trainer._lambda_thin_plate_energy_ema = (
                            (1.0 - alpha_thin) * Trainer._lambda_thin_plate_energy_ema
                            + alpha_thin * raw_lambda_thin
                        )

                        lambda_thin_plate_energy_cur = Trainer._lambda_thin_plate_energy_ema

                        # warmup ramp
                        t = min((iteration - warmup_start) / warmup_len, 1.0)
                        lambda_thin_plate_energy_cur *= t

                    thinplate_loss = lambda_thin_plate_energy_cur * thinplate_loss_raw
                    total_loss = total_loss + thinplate_loss

                    loss_dict['thinPlateE_raw'] = thinplate_loss_raw.item()
                    loss_dict['thinPlateE'] = thinplate_loss.item()
                    loss_dict['lambda_thinPlate'] = float(lambda_thin_plate_energy_cur)

                # ----------------------------
                # 2) SDF Hessian regularizer
                # ----------------------------
                if target_ratio_hes > 0:
                # if False: 
                    x_nx3 = fc_params['x_nx3']
                    hessian_loss_raw = sdf_hessian_energy_loss(sdf, grid_edges, x_nx3)

                    if iteration < warmup_start:
                        lambda_smooth_cur = 0.0
                    else:
                        raw_lambda_hes = target_ratio_hes * render_ref / (hessian_loss_raw.detach() + 1e-8)

                        if not hasattr(Trainer, "_lambda_hessian_energy_ema"):
                            Trainer._lambda_hessian_energy_ema = raw_lambda_hes

                        alpha_hes = 0.02
                        Trainer._lambda_hessian_energy_ema = (
                            (1.0 - alpha_hes) * Trainer._lambda_hessian_energy_ema
                            + alpha_hes * raw_lambda_hes
                        )

                        lambda_smooth_cur = Trainer._lambda_hessian_energy_ema

                        # warmup ramp
                        t = min((iteration - warmup_start) / warmup_len, 1.0)
                        lambda_smooth_cur *= t

                    loss_hessian = lambda_smooth_cur * hessian_loss_raw
                    total_loss = total_loss + loss_hessian

                    loss_dict['hes_raw'] = hessian_loss_raw.item()
                    loss_dict['hes'] = loss_hessian.item()
                    loss_dict['lambda_hes'] = float(lambda_smooth_cur)


            # FlexiCubes regularizers 
            if lambda_reg > 0: 
                ## copied from flexicube examples. 
                t_iter = iteration / num_iterations 
                sdf_weight = lambda_reg - (lambda_reg - lambda_reg/20)*min(1.0, 4.0 * t_iter)

                reg_loss = flexicube_sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight
                reg_loss = reg_loss + L_dev.mean() * 0.5 

                weight = fc_params['weight'] 
                reg_loss = reg_loss + (weight[:,:20]).abs().mean() * 0.1 

                loss_dict['Reg'] = reg_loss.item() 
                total_loss = total_loss + reg_loss 
            


            # 检查损失是否包含 NaN 或 Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f'[WARNING] Invalid loss (NaN/Inf) at iteration {iteration}, skipping...')
                continue

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [fc_params['sdf'], fc_params['deform'], fc_params['weight']],
                max_norm=1.0
            )

            # 检查梯度是否包含 NaN 或 Inf
            has_nan_grad = False
            has_inf_grad = False
            for param in [fc_params['sdf'], fc_params['deform'], fc_params['weight']]:
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
            optimizer.step()

            # 裁剪 SDF 值
            with torch.no_grad():
                fc_params['sdf'].data = torch.clamp(fc_params['sdf'].data, -10.0, 10.0)

            if writer is not None:
                writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                # 记录所有实际计算的 loss
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'Loss/{loss_name}', loss_value, iteration)

            # 更新进度条和日志
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                if writer is not None:
                    # 记录渲染图像
                    for i, (render_data_group, render_idx) in enumerate(zip(render_data_list, render_idx_list)):
                        for j, render_data in enumerate(render_data_group): 
                            if render_data.dim() == 3 and render_data.shape[-1] == 3:
                                render_data = render_data.permute(2, 0, 1)
                            writer.add_image(f'Render/Camera_{i}_{j}', render_data, global_step=iteration)

            # 更新进度条（只显示实际使用的 loss）
            postfix_dict = {'loss': f'{total_loss.item():.4f}'}
            if 'Render' in loss_dict:
                postfix_dict['render'] = f'{loss_dict["Render"]:.4f}'
            if 'Reg' in loss_dict:
                postfix_dict['reg'] = f'{loss_dict["Reg"]:.6f}' 
            if 'hes' in loss_dict: 
                postfix_dict['hes'] = f'{loss_dict["hes"]:.6f}' 
            if 'thinPlateE' in loss_dict: 
                postfix_dict['tp'] = f'{loss_dict["thinPlateE"]:.6f}' 

            pbar.set_postfix(postfix_dict)

        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()

        # 提取最终 mesh
        final_mesh, final_vertices, _, _, verts_sh_coeff = FCConvertor.extractMesh(fc_params, training=True)
        if verts_sh_coeff is not None: 
            final_mesh = bake_vertex_colors_from_sh(
                renderer=renderer,
                camera_list=camera_list,
                final_mesh=final_mesh,
                final_vertices=final_vertices,
                verts_sh_coeff=verts_sh_coeff,
                sh_deg=fc_params["sh_deg"],
            )
        return final_mesh
