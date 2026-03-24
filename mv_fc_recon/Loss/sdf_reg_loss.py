import torch


def flexicube_sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff




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