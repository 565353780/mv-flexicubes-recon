import torch
import numpy as np
from flexi_cubes.Module.sh_utils import RGB2SH, SH2RGB, eval_sh 

def bake_vertex_colors_from_sh(
    renderer,
    camera_list,
    final_mesh,
    final_vertices,
    verts_sh_coeff,
    sh_deg,
    default_color: float = 1,
):
    """
    根据多视角 mask，可见性筛选顶点，并把 SH 颜色烘焙到 final_mesh.visual.vertex_colors

    Args:
        renderer: 你的渲染器，需支持 render(camera=..., vertices=..., faces=..., renderTypes=["mask"])
        camera_list: 相机列表。每个 camera 需要有:
            - camera.pos
            - camera.mask
        final_mesh: trimesh mesh
        final_vertices: [V, 3] torch.Tensor
        verts_sh_coeff: SH 系数
        sh_deg: SH 阶数
        default_color: 没有被任何视角看到的顶点默认颜色

    Returns:
        final_mesh
    """
    
    device = final_vertices.device
    faces = torch.from_numpy(final_mesh.faces).long().to(device)   # [F, 3]

    V = final_vertices.shape[0]
    accum_rgb = torch.zeros((V, 3), device=device, dtype=torch.float32)
    accum_w = torch.zeros((V, 1), device=device, dtype=torch.float32)

    for camera in camera_list:
        render_out = renderer.render(
            final_mesh, 
            camera=camera,
            vertices_tensor=final_vertices,   # [1, V, 3]
            return_types=["mask"]
        )

        mask_dict = render_out["mask"]
        rast_mask = mask_dict["rast_mask"]            # [H, W], bool
        rast_face_idx = mask_dict["rast_face_idx"]    # [H, W], float, triangle_id+1

        cam_mask = camera.mask
        if not torch.is_tensor(cam_mask):
            cam_mask = torch.from_numpy(cam_mask)
        cam_mask = cam_mask.to(device)

        if cam_mask.ndim == 3 and cam_mask.shape[-1] == 1:
            cam_mask = cam_mask[..., 0]

        cam_mask = cam_mask > 0.5
        valid_pixel_mask = rast_mask & cam_mask

        hit_face_ids = rast_face_idx[valid_pixel_mask].long() - 1
        hit_face_ids = hit_face_ids[hit_face_ids >= 0]

        if hit_face_ids.numel() == 0:
            continue

        hit_face_ids = torch.unique(hit_face_ids)
        hit_vertex_ids = torch.unique(faces[hit_face_ids].reshape(-1))

        if hit_vertex_ids.numel() == 0:
            continue

        view_dirs = final_vertices - camera.pos
        out_rgb = eval_sh(sh_deg, verts_sh_coeff, view_dirs)
        out_rgb = SH2RGB(out_rgb)
        out_rgb = torch.clamp(out_rgb, 0, 1)

        accum_rgb[hit_vertex_ids] += out_rgb[hit_vertex_ids]
        accum_w[hit_vertex_ids] += 1.0

    valid_v = accum_w.squeeze(-1) > 0
    baked_rgb = torch.zeros_like(accum_rgb)
    baked_rgb[valid_v] = accum_rgb[valid_v] / accum_w[valid_v]
    baked_rgb[~valid_v] = default_color

    vertex_colors = (baked_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    final_mesh.visual.vertex_colors = vertex_colors

    return final_mesh