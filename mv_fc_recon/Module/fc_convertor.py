import os
import cv2
import torch
import kaolin
import pickle
import trimesh
from typing import Union, List
from kaolin.ops.conversions import trianglemesh_to_voxelgrid

from camera_control.Module.rgbd_camera import RGBDCamera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mv_fc_recon.Method.io import loadMeshFile

def demo_render(mesh: trimesh.Trimesh, bg_color=[255, 255, 255]):

class FCConvertor(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createFC(
        mesh: Union[str, trimesh.Trimesh],
        resolution: int=128,
        device: str='cuda:0',
    ):
        if isinstance(mesh, str):
            if not os.path.exists(mesh):
                print('[ERROR][FCConvertor::createFC]')
                print('\t mesh file not exist!')
                print('\t mesh_file_path:', mesh)
                return None

            mesh = loadMeshFile(mesh)

        fc = kaolin.ops.flexicubes.FlexiCubes(device=device)
        return fc

    @staticmethod
    def fitImages(
        camera_list: List[RGBDCamera],
        mesh: Union[str, trimesh.Trimesh],
        resolution: int=128,
        device: str='cuda:0',
        bg_color: list=[255, 255, 255],
    ) -> trimesh.Trimesh:
        fc = FCConvertor.createFC(mesh, resolution, device)

        for camera in camera_list:
            camera.to(device=device)

        # 你的可学习参数 (SDF值 + 变形权重)
        # sdf: [Batch, Res, Res, Res]
        # deform: [Batch, Res, Res, Res, 3] -> 允许网格顶点微调
        sdf = torch.nn.Parameter(initial_sdf, requires_grad=True) 
        deform = torch.nn.Parameter(torch.zeros_like(initial_sdf_vec), requires_grad=True)
        weight = torch.nn.Parameter(torch.zeros(...), requires_grad=True) # FlexiCubes 特有的权重

        optimizer = torch.optim.Adam([sdf, deform, weight], lr=0.01)

        # 2. 训练循环
        for i in range(1000):
            for camera in camera_list:
                # --- 关键步骤：可微提取 Mesh ---
                # 输出的 vertices 和 faces 是可导的！
                vertices, faces, L_dev = fc(
                    sdf, 
                    weight, 
                    voxel_resolution=resolution,
                )

                mesh = trimesh.Trimesh(
                    vertices=vertices.detach().clone().cpu().numpy(),
                    faces=faces.cpu().numpy(),
                )

                render_texture_dict = NVDiffRastRenderer.renderTexture(
                    mesh=mesh,
                    camera=camera_list[0],
                    bg_color=bg_color,
                    vertices_tensor=vertices,
                )

                render_normal_dict = NVDiffRastRenderer.renderNormal(
                    mesh=mesh,
                    camera=camera_list[0],
                    bg_color=bg_color,
                    vertices_tensor=vertices,
                )

                os.makedirs('./output/', exist_ok=True)

                cv2.imwrite('./output/test_render_texture.png', render_texture_dict['image'])
                cv2.imwrite('./output/test_render_normal_camera.png', render_normal_dict['normal_camera'])
                cv2.imwrite('./output/test_render_normal_world.png', render_normal_dict['normal_world'])

                loss_reg = kaolin.ops.flexicubes.regularize_weights(weight, sdf)

                (loss_render + loss_reg).backward()
        return
