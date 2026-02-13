import torch


def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff





def sdf_smoothness_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    mode: str = 'adaptive',
    threshold: float = 0.1,
) -> torch.Tensor:
    """SDF 平滑正则化：惩罚相邻网格点 SDF 差异过大

    这比 Eikonal loss 更适合 FlexiCubes，因为它：
    1. 不强制梯度模长为 1
    2. 只惩罚局部不平滑，允许 SDF 有任意尺度

    Args:
        sdf: [N] SDF 网格值
        grid_edges: [E, 2] 边索引
        mode: 平滑模式
            - 'l2': 简单 L2 平滑，惩罚所有差异
            - 'adaptive': 自适应平滑，只惩罚超过阈值的差异
            - 'huber': Huber loss，对大差异使用 L1，小差异使用 L2
        threshold: adaptive/huber 模式的阈值

    Returns:
        loss: 平滑损失
    """
    # 获取边两端的 SDF 值
    sdf_edge = sdf[grid_edges]  # [E, 2]
    diff = sdf_edge[:, 0] - sdf_edge[:, 1]  # [E]

    if mode == 'l2':
        # 简单 L2 平滑
        loss = (diff ** 2).mean()

    elif mode == 'adaptive':
        # 自适应平滑：只惩罚超过阈值的差异
        # 这允许 SDF 有一定的局部变化（用于捕捉几何细节）
        # 但惩罚过大的变化（噪声）
        diff_abs = diff.abs()
        excess = torch.relu(diff_abs - threshold)
        loss = (excess ** 2).mean()

    elif mode == 'huber':
        # Huber loss：对大差异使用 L1（线性），小差异使用 L2（二次）
        # 这对异常值（噪声）更鲁棒
        diff_abs = diff.abs()
        quadratic = torch.clamp(diff_abs, max=threshold)
        linear = diff_abs - quadratic
        loss = (0.5 * quadratic ** 2 + threshold * linear).mean()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return loss


def sdf_hessian_energy_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    x_nx3: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Hessian energy loss using centered differences for mixed derivatives.
    """
    # Determine grid dimensions from coordinates
    unique_x = torch.unique(x_nx3[:, 0], sorted=True)
    unique_y = torch.unique(x_nx3[:, 1], sorted=True)
    unique_z = torch.unique(x_nx3[:, 2], sorted=True)
    
    Nx = len(unique_x)
    Ny = len(unique_y)
    Nz = len(unique_z)
    
    assert Nx * Ny * Nz == len(sdf), "Grid dimensions don't match number of points"
    
    # Compute grid spacing
    dx = unique_x[1] - unique_x[0] if Nx > 1 else 1.0
    dy = unique_y[1] - unique_y[0] if Ny > 1 else 1.0
    dz = unique_z[1] - unique_z[0] if Nz > 1 else 1.0
    
    dV = dx * dy * dz
    
    # Reshape sdf to 3D grid
    sdf_3d = sdf.reshape(Nx, Ny, Nz)
    
    # Initialize tensors for second derivatives
    fxx = torch.zeros_like(sdf_3d)
    fyy = torch.zeros_like(sdf_3d)
    fzz = torch.zeros_like(sdf_3d)
    fxy = torch.zeros_like(sdf_3d)
    fxz = torch.zeros_like(sdf_3d)
    fyz = torch.zeros_like(sdf_3d)
    
    # Pure second derivatives (central difference)
    if Nx > 2:
        fxx[1:-1, :, :] = (sdf_3d[2:, :, :] - 2 * sdf_3d[1:-1, :, :] + sdf_3d[:-2, :, :]) / (dx**2)
    if Ny > 2:
        fyy[:, 1:-1, :] = (sdf_3d[:, 2:, :] - 2 * sdf_3d[:, 1:-1, :] + sdf_3d[:, :-2, :]) / (dy**2)
    if Nz > 2:
        fzz[:, :, 1:-1] = (sdf_3d[:, :, 2:] - 2 * sdf_3d[:, :, 1:-1] + sdf_3d[:, :, :-2]) / (dz**2)
    
    # Mixed derivatives (central difference)
    if Nx > 2 and Ny > 2:
        # fxy = ∂²f/∂x∂y using central differences
        fxy[1:-1, 1:-1, :] = (sdf_3d[2:, 2:, :] - sdf_3d[2:, :-2, :] - 
                               sdf_3d[:-2, 2:, :] + sdf_3d[:-2, :-2, :]) / (4 * dx * dy)
    
    if Nx > 2 and Nz > 2:
        # fxz = ∂²f/∂x∂z using central differences
        fxz[1:-1, :, 1:-1] = (sdf_3d[2:, :, 2:] - sdf_3d[2:, :, :-2] - 
                               sdf_3d[:-2, :, 2:] + sdf_3d[:-2, :, :-2]) / (4 * dx * dz)
    
    if Ny > 2 and Nz > 2:
        # fyz = ∂²f/∂y∂z using central differences
        fyz[:, 1:-1, 1:-1] = (sdf_3d[:, 2:, 2:] - sdf_3d[:, 2:, :-2] - 
                               sdf_3d[:, :-2, 2:] + sdf_3d[:, :-2, :-2]) / (4 * dy * dz)
    
    # Compute the integrand
    integrand = (fxx**2 + fyy**2 + fzz**2 + 
                 2 * fxy**2 + 2 * fxz**2 + 2 * fyz**2)
    
    loss = torch.mean(integrand) * dV
    
    return loss



def sdf_hessian_energy_loss_accurate(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    x_nx3: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Hessian energy (smoothness) loss for a scalar field on a regular grid.
    
    Loss = ∫(fxx² + fyy² + fzz² + 2fxy² + 2fxz² + 2fyz²) dV
    
    Args:
        sdf: [N] tensor of scalar values on regular grid points
        grid_edges: [E, 2] grid edge indices - each row contains [i, j] for edge from point i to j
        x_nx3: [N, 3] tensor of grid point coordinates
    
    Returns:
        loss: scalar tensor representing the discretized Hessian energy
    """
    # Determine grid dimensions from coordinates
    unique_x = torch.unique(x_nx3[:, 0], sorted=True)
    unique_y = torch.unique(x_nx3[:, 1], sorted=True)
    unique_z = torch.unique(x_nx3[:, 2], sorted=True)
    
    Nx = len(unique_x)
    Ny = len(unique_y)
    Nz = len(unique_z)
    
    # Compute grid spacing
    dx = unique_x[1] - unique_x[0] if Nx > 1 else 1.0
    dy = unique_y[1] - unique_y[0] if Ny > 1 else 1.0
    dz = unique_z[1] - unique_z[0] if Nz > 1 else 1.0
    
    # Cell volume
    dV = dx * dy * dz
    
    # Reshape sdf to 3D grid for easier indexing
    sdf_3d = sdf.reshape(Nx, Ny, Nz)
    
    # Identify edges along each direction using coordinate differences
    edge_vectors = x_nx3[grid_edges[:, 1]] - x_nx3[grid_edges[:, 0]]
    
    # Find edges aligned with each axis (tolerance for floating point)
    eps = 1e-6
    edges_x = torch.abs(edge_vectors[:, 1]) < eps and torch.abs(edge_vectors[:, 2]) < eps
    edges_y = torch.abs(edge_vectors[:, 0]) < eps and torch.abs(edge_vectors[:, 2]) < eps
    edges_z = torch.abs(edge_vectors[:, 0]) < eps and torch.abs(edge_vectors[:, 1]) < eps
    
    # Compute first derivatives using edges
    # For each direction, we can directly compute the derivative at edge midpoints
    grad_x_mid = torch.zeros(len(grid_edges), device=sdf.device)
    grad_y_mid = torch.zeros(len(grid_edges), device=sdf.device)
    grad_z_mid = torch.zeros(len(grid_edges), device=sdf.device)
    
    # X-direction edges
    if torch.any(edges_x):
        edge_indices = torch.where(edges_x)[0]
        for idx in edge_indices:
            i0, i1 = grid_edges[idx]
            grad_x_mid[idx] = (sdf[i1] - sdf[i0]) / dx
    
    # Y-direction edges
    if torch.any(edges_y):
        edge_indices = torch.where(edges_y)[0]
        for idx in edge_indices:
            i0, i1 = grid_edges[idx]
            grad_y_mid[idx] = (sdf[i1] - sdf[i0]) / dy
    
    # Z-direction edges
    if torch.any(edges_z):
        edge_indices = torch.where(edges_z)[0]
        for idx in edge_indices:
            i0, i1 = grid_edges[idx]
            grad_z_mid[idx] = (sdf[i1] - sdf[i0]) / dz
    
    # Now compute second derivatives using pairs of parallel edges
    fxx = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    fyy = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    fzz = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    fxy = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    fxz = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    fyz = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    
    count_xx = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    count_yy = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    count_zz = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    count_xy = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    count_xz = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    count_yz = torch.zeros(Nx, Ny, Nz, device=sdf.device)
    
    # For fxx: use pairs of x-edges that share the same y,z coordinates
    if torch.any(edges_x):
        # Group x-edges by their (y,z) coordinates
        x_edge_coords = []
        x_edge_values = []
        x_edge_indices = torch.where(edges_x)[0]
        
        for idx in x_edge_indices:
            i0, i1 = grid_edges[idx]
            # Get coordinates of the first endpoint
            coord = x_nx3[i0]
            # For x-edge, we store (y,z) as key
            x_edge_coords.append((coord[1].item(), coord[2].item()))
            x_edge_values.append(grad_x_mid[idx])
        
        # Group edges by (y,z)
        # For each unique (y,z), collect all x-edges and sort by x-coordinate
        # This is simplified - in practice you'd need a more efficient grouping
    
    # For simplicity and correctness, let's use finite differences on the grid
    # This is more straightforward and still valid for regular grids
    
    # Pure second derivatives using central differences
    if Nx > 2:
        fxx[1:-1, :, :] = (sdf_3d[2:, :, :] - 2 * sdf_3d[1:-1, :, :] + sdf_3d[:-2, :, :]) / (dx**2)
        count_xx[1:-1, :, :] = 1
    
    if Ny > 2:
        fyy[:, 1:-1, :] = (sdf_3d[:, 2:, :] - 2 * sdf_3d[:, 1:-1, :] + sdf_3d[:, :-2, :]) / (dy**2)
        count_yy[:, 1:-1, :] = 1
    
    if Nz > 2:
        fzz[:, :, 1:-1] = (sdf_3d[:, :, 2:] - 2 * sdf_3d[:, :, 1:-1] + sdf_3d[:, :, :-2]) / (dz**2)
        count_zz[:, :, 1:-1] = 1
    
    # Mixed derivatives using central differences
    if Nx > 1 and Ny > 1:
        # fxy at cell centers
        fxy_centers = (sdf_3d[1:, 1:, :] - sdf_3d[1:, :-1, :] - 
                       sdf_3d[:-1, 1:, :] + sdf_3d[:-1, :-1, :]) / (dx * dy)
        # Distribute to vertices (averaging 4 adjacent cells)
        fxy[:-1, :-1, :] += fxy_centers
        fxy[1:, :-1, :] += fxy_centers
        fxy[:-1, 1:, :] += fxy_centers
        fxy[1:, 1:, :] += fxy_centers
        count_xy[:-1, :-1, :] += 1
        count_xy[1:, :-1, :] += 1
        count_xy[:-1, 1:, :] += 1
        count_xy[1:, 1:, :] += 1
    
    if Nx > 1 and Nz > 1:
        # fxz at cell centers
        fxz_centers = (sdf_3d[1:, :, 1:] - sdf_3d[1:, :, :-1] - 
                       sdf_3d[:-1, :, 1:] + sdf_3d[:-1, :, :-1]) / (dx * dz)
        # Distribute to vertices
        fxz[:-1, :, :-1] += fxz_centers
        fxz[1:, :, :-1] += fxz_centers
        fxz[:-1, :, 1:] += fxz_centers
        fxz[1:, :, 1:] += fxz_centers
        count_xz[:-1, :, :-1] += 1
        count_xz[1:, :, :-1] += 1
        count_xz[:-1, :, 1:] += 1
        count_xz[1:, :, 1:] += 1
    
    if Ny > 1 and Nz > 1:
        # fyz at cell centers
        fyz_centers = (sdf_3d[:, 1:, 1:] - sdf_3d[:, 1:, :-1] - 
                       sdf_3d[:, :-1, 1:] + sdf_3d[:, :-1, :-1]) / (dy * dz)
        # Distribute to vertices
        fyz[:, :-1, :-1] += fyz_centers
        fyz[:, 1:, :-1] += fyz_centers
        fyz[:, :-1, 1:] += fyz_centers
        fyz[:, 1:, 1:] += fyz_centers
        count_yz[:, :-1, :-1] += 1
        count_yz[:, 1:, :-1] += 1
        count_yz[:, :-1, 1:] += 1
        count_yz[:, 1:, 1:] += 1
    
    # Average the mixed derivatives
    fxy = fxy / (count_xy + 1e-8)
    fxz = fxz / (count_xz + 1e-8)
    fyz = fyz / (count_yz + 1e-8)
    
    # Compute the integrand
    integrand = (fxx**2 + fyy**2 + fzz**2 + 
                 2 * fxy**2 + 2 * fxz**2 + 2 * fyz**2)
    
    # Integrate over the volume
    loss = torch.sum(integrand) * dV
    
    return loss













def sdf_gradient_smoothness_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    x_nx3: torch.Tensor,
    mode: str = 'local',
) -> torch.Tensor:
    """SDF 梯度平滑正则化：惩罚梯度变化过大（二阶平滑）

    与 Eikonal 不同，这里不强制梯度模长为 1，
    而是惩罚梯度在空间上的变化。

    这对于保持几何细节同时抑制高频噪声非常有效：
    - 一阶平滑（sdf_smoothness_loss）会抹平所有细节
    - 二阶平滑只惩罚"梯度的变化"，允许平滑的斜坡但抑制尖锐振荡

    Args:
        sdf: [N] SDF 网格值
        grid_edges: [E, 2] 边索引
        x_nx3: [N, 3] 网格顶点坐标
        mode: 平滑模式
            - 'global': 惩罚梯度偏离全局均值（原始实现）
            - 'local': 惩罚相邻边梯度差异（更好地保持细节）

    Returns:
        loss: 梯度平滑损失
    """
    # 计算每条边上的 SDF 梯度（近似）
    sdf_edge = sdf[grid_edges]  # [E, 2]
    pos_edge = x_nx3[grid_edges]  # [E, 2, 3]

    # 边方向和长度
    edge_vec = pos_edge[:, 1] - pos_edge[:, 0]  # [E, 3]
    edge_len = edge_vec.norm(dim=-1).clamp(min=1e-8)  # [E]

    # SDF 沿边方向的梯度
    grad_along_edge = (sdf_edge[:, 1] - sdf_edge[:, 0]) / edge_len  # [E]

    if mode == 'global':
        # 惩罚梯度偏离全局均值
        grad_mean = grad_along_edge.mean()
        loss = ((grad_along_edge - grad_mean) ** 2).mean()

    elif mode == 'local':
        # 惩罚相邻顶点的梯度差异
        # 为每个顶点计算其所有出边的平均梯度
        device = sdf.device
        num_vertices = sdf.shape[0]

        # 累加每个顶点的梯度和
        grad_sum = torch.zeros(num_vertices, device=device)
        grad_count = torch.zeros(num_vertices, device=device)

        # 边的起点和终点
        src_idx = grid_edges[:, 0]  # [E]
        dst_idx = grid_edges[:, 1]  # [E]

        # 对于每条边，将梯度累加到两个端点
        grad_sum.scatter_add_(0, src_idx, grad_along_edge)
        grad_sum.scatter_add_(0, dst_idx, -grad_along_edge)  # 反向边梯度取反
        grad_count.scatter_add_(0, src_idx, torch.ones_like(grad_along_edge))
        grad_count.scatter_add_(0, dst_idx, torch.ones_like(grad_along_edge))

        # 避免除以 0
        grad_count = grad_count.clamp(min=1)

        # 每个顶点的平均梯度
        vertex_grad_mean = grad_sum / grad_count  # [N]

        # 计算每条边的梯度与其端点平均梯度的差异
        src_grad_mean = vertex_grad_mean[src_idx]  # [E]
        dst_grad_mean = vertex_grad_mean[dst_idx]  # [E]

        # 梯度应该接近端点的平均值
        diff_src = (grad_along_edge - src_grad_mean) ** 2
        diff_dst = (-grad_along_edge - dst_grad_mean) ** 2

        loss = (diff_src + diff_dst).mean() / 2

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return loss


def sdf_sign_consistency_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """SDF 符号一致性损失：软化版的 SDF 正则化

    原始的 BCE 版本在 SDF 接近 0 时梯度爆炸。
    这个版本使用更稳定的 margin-based loss。

    对于符号变化的边（表面穿过的边），鼓励两端 SDF 值接近 0。
    对于符号相同的边，不施加约束。

    Args:
        sdf: [N] SDF 网格值
        grid_edges: [E, 2] 边索引
        margin: SDF 值的 margin

    Returns:
        loss: 符号一致性损失
    """
    sdf_edge = sdf[grid_edges]  # [E, 2]

    # 找到符号变化的边（表面穿过的边）
    sign_change = (sdf_edge[:, 0] * sdf_edge[:, 1]) < 0  # [E]

    if sign_change.sum() == 0:
        return torch.tensor(0.0, device=sdf.device)

    # 对于符号变化的边，鼓励两端 SDF 绝对值较小
    # 这样表面位置更精确
    sdf_abs = sdf_edge[sign_change].abs()  # [E', 2]

    # 使用 soft margin loss：max(|sdf| - margin, 0)^2
    # 这比 BCE 更稳定
    excess = torch.relu(sdf_abs - margin)
    loss = (excess ** 2).mean()

    return loss


def weight_regularization_loss(
    weight: torch.Tensor,
    target_scale: float = 0.5,
) -> torch.Tensor:
    """FlexiCubes 权重正则化

    约束 alpha, beta, gamma 权重不要偏离太远，
    防止某些立方体使用极端权重导致表面扭曲。

    Args:
        weight: [F, 21] FlexiCubes 权重
            - [:, :12]: beta (12 条边的插值权重)
            - [:, 12:20]: alpha (8 个顶点的权重)
            - [:, 20]: gamma_f (每个立方体的权重)
        target_scale: 目标缩放因子

    Returns:
        loss: 权重正则化损失
    """
    # 分离不同类型的权重
    beta = weight[:, :12]  # [F, 12]
    alpha = weight[:, 12:20]  # [F, 8]
    gamma = weight[:, 20]  # [F]

    # L2 正则化，鼓励权重接近 0（FlexiCubes 默认行为）
    loss_beta = (beta ** 2).mean()
    loss_alpha = (alpha ** 2).mean()
    loss_gamma = (gamma ** 2).mean()

    return loss_beta + loss_alpha + loss_gamma



def short_edge_loss(
    V: torch.Tensor,
    F: torch.Tensor,
    target_ratio: float = 0.5,
    strength: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    惩罚网格中相对于平均边长过短的边。

    Args:
        V: [num_vertices, 3] 顶点坐标
        F: [num_faces, 3] 面片索引
        target_ratio: 小于平均边长 * target_ratio 的边会被惩罚
        strength: loss 权重系数
        eps: 防止除零

    Returns:
        loss: 标量张量
    """
    # 提取每条边
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e0 = torch.norm(v1 - v2, dim=1)
    e1 = torch.norm(v2 - v0, dim=1)
    e2 = torch.norm(v0 - v1, dim=1)

    lengths = torch.cat([e0, e1, e2], dim=0)

    # 计算平均边长
    mean_len = lengths.mean()

    # 最小边阈值 = 平均边长 * target_ratio
    min_len = mean_len * target_ratio

    # 计算比率差异: ratio = edge_length / min_len
    ratio = lengths / (min_len + eps)

    # 只惩罚 ratio < 1 的边
    short_edges = torch.relu(1.0 - ratio)

    # loss = 平均 ( (1 - ratio)^2 )
    loss = (short_edges ** 2).mean() * strength

    return loss




def mesh_normal_consistency_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    法线一致性损失（与叉乘方向无关，向量化实现）

    惩罚相邻面片法线差异过大，用于平滑表面。

    Args:
        vertices: [V, 3] 网格顶点
        faces: [F, 3] 网格面片

    Returns:
        loss: 法线一致性损失
    """
    device = vertices.device
    F_num = faces.shape[0]

    # --- 计算面片法线 ---
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = face_normals / face_normals.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [F,3]

    # --- 构建共享边的邻接对 ---
    # edges: [3*F, 2]
    edges = torch.cat([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]], dim=0)
    # edge -> faces mapping
    faces_idx = torch.arange(F_num, device=device).repeat(3)

    # 为了向量化，先把 edges 排序
    edges_sorted, _ = torch.sort(edges, dim=1)

    # 使用 unique 找到重复边（共享边）
    edges_unique, inverse, counts = torch.unique(edges_sorted, return_inverse=True, return_counts=True, dim=0)

    # 只保留共享边
    shared_edge_mask = counts[inverse] == 2
    edges_shared = edges_sorted[shared_edge_mask]
    faces_shared = faces_idx[shared_edge_mask]

    # 每对边对应的两个面
    # 分组为 shape [P,2]，P = number of shared edges
    perm = torch.argsort(inverse[shared_edge_mask])
    faces_pairs = faces_shared[perm].view(-1, 2)

    if faces_pairs.numel() == 0:
        return torch.tensor(0.0, device=device)

    # --- 计算相邻面片法线差异 ---
    n1 = face_normals[faces_pairs[:, 0]]
    n2 = face_normals[faces_pairs[:, 1]]

    # 使用绝对值避免叉乘顺序/法线方向问题
    dot = (n1 * n2).sum(dim=1).clamp(-1.0, 1.0)
    loss = (1.0 - dot.abs()).mean()

    return loss


