"""
Geometry utilities for GMFlow
"""
import torch
import torch.nn.functional as F


def coords_grid(b, h, w, device):
    """
    Generate coordinate grid
    Args:
        b: batch size
        h: height
        w: width
        device: torch device
    Returns:
        coords: [B, 2, H, W]
    """
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(b, 1, 1, 1)


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    """
    Wrapper for grid_sample with pixel coordinates
    Args:
        img: [B, C, H, W]
        sample_coords: [B, 2, H, W] in pixel coordinates
        mode: interpolation mode
        padding_mode: padding mode for outside grid values
        return_mask: if True, return mask for valid samples
    Returns:
        sampled_img: [B, C, H, W]
        mask (optional): [B, 1, H, W]
    """
    b, c, h, w = img.shape

    # Normalize coordinates to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (x_grid <= 1) & (y_grid >= -1) & (y_grid <= 1)  # [B, H, W]
        mask = mask.float().unsqueeze(1)  # [B, 1, H, W]
        return img, mask

    return img


def flow_warp(feature, flow, padding_mode='zeros'):
    """
    Warp feature map according to flow
    Args:
        feature: [B, C, H, W]
        flow: [B, 2, H, W]
        padding_mode: padding mode for grid sample
    Returns:
        warped_feature: [B, C, H, W]
    """
    b, c, h, w = feature.size()

    # Generate base coordinate grid
    coords = coords_grid(b, h, w, device=flow.device)  # [B, 2, H, W]

    # Add flow to coordinates
    coords = coords + flow  # [B, 2, H, W]

    # Warp using bilinear sampling
    warped_feature = bilinear_sample(feature, coords, padding_mode=padding_mode)

    return warped_feature


def compute_flow_magnitude(flow):
    """
    Compute flow magnitude
    Args:
        flow: [B, 2, H, W]
    Returns:
        magnitude: [B, 1, H, W]
    """
    magnitude = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2)
    return magnitude
