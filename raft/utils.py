"""
Utility functions for RAFT
"""
import torch
import torch.nn.functional as F


def coords_grid(batch, ht, wd, device):
    """
    Generate coordinate grid
    Args:
        batch: batch size
        ht: height
        wd: width
        device: torch device
    Returns:
        coords: [B, 2, H, W]
    """
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    """
    Upsample flow by factor of 8
    Args:
        flow: [B, 2, H, W]
        mode: interpolation mode
    Returns:
        upsampled flow: [B, 2, H*8, W*8]
    """
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """
    Wrapper for grid_sample, uses pixel coordinates
    Args:
        img: [B, C, H, W]
        coords: [B, 2, H, W] pixel coordinates
        mode: interpolation mode
        mask: if True, return mask for valid samples
    Returns:
        sampled_img: [B, C, H, W]
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=1).permute(0, 2, 3, 1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask_val = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask_val.float()

    return img
