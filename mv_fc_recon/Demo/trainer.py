import sys
sys.path.append('../camera-control')
sys.path.append('../flexi-cubes')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import pickle

from camera_control.Method.io import loadMeshFile
from camera_control.Method.mesh import normalizeMesh
from camera_control.Module.mesh_renderer import MeshRenderer
# from camera_control.Module.mesh_rendererV2 import MeshRenderer

from mv_fc_recon.Method.time import getCurrentTime
# from mv_fc_recon.Module.trainer import Trainer
from mv_fc_recon.Module.trainer_refact import Trainer, TrainerConfig



def demo():
    # 小妖怪头
    # shape_id = "003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
    # 女人上半身
    # shape_id = "017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
    # 长发男人头
    shape_id = "0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"
    # 带texture的牛
    # shape_id = "spot"
    # shape_id = "00ba54f467444f93bb58ebf819e0277f5260b13016c8cce5de39d69e2b9d3e86" 

    device = 'cuda:0'

    home = os.environ['HOME']

    data_folder_path = home + "/chLi/Dataset/pixel_align/" + shape_id + "/"

    camera_pkl_file_path = data_folder_path + 'camera_fc.pkl'
    # if not os.path.exists(camera_pkl_file_path):
    if True: 
        normalized_gt_mesh_file_path = data_folder_path + 'gt_normalized.ply'
        # normalized_gt_mesh_file_path = data_folder_path + 'gt_normalized.glb'
        # if not os.path.exists(normalized_gt_mesh_file_path):
        if True: 
            gt_mesh_file_path = data_folder_path + 'gt.glb'
            mesh = loadMeshFile(gt_mesh_file_path)
            mesh = normalizeMesh(mesh, target_length=0.99)
            mesh.export(normalized_gt_mesh_file_path)

        mesh = loadMeshFile(normalized_gt_mesh_file_path)

        camera_list = MeshRenderer.sampleRenderData(
            mesh,
            camera_num=50,
            camera_dist_range=[1.4, 1.4],
            width=1024,
            height=1024,
            fovx_degree_range=[55, 55],
            device=device,
        )
        with open(camera_pkl_file_path, "wb") as f:
            pickle.dump(camera_list, f)

    gen_mesh_file_path = data_folder_path + "stage2_full.glb"
    # gen_mesh_file_path = data_folder_path + "spot_sharp.glb" 
    log_dir = data_folder_path + 'mv-fc-recon/logs/' + "refact_loss" + '/'

    assert os.path.exists(camera_pkl_file_path), f"camera.pkl not found at {camera_pkl_file_path}"
    with open(camera_pkl_file_path, 'rb') as f:
        camera_list = pickle.load(f)

    mesh = loadMeshFile(gen_mesh_file_path)
    mesh = normalizeMesh(mesh, target_length=0.99)
    # fitting_mesh = Trainer.fitImagesWithSDFLoss(
    #     camera_list=camera_list,
    #     mesh=mesh,
    #     resolution=192,
    #     device=device,
    #     num_iterations=300,
    #     log_interval=10,
    #     log_dir=log_dir,
    # )
    trainer = Trainer(TrainerConfig(
        camera_list=camera_list,
        initMesh=mesh,
        resolution=192,
        device=device,
        num_iterations=300,
        log_interval=10,
        log_dir=log_dir
    )) 
    fitting_mesh = trainer.fit()

    fitting_mesh.export(log_dir + 'fc_fitting_mesh.ply')
    print("[result] ", shape_id + '_fc_fitting_mesh.ply') 
    return True
