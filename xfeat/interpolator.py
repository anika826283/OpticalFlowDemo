"""
Bilinear interpolation utilities for XFeat
"""
import torch
import torch.nn.functional as F


def interpolate_dense(keypoints, dense):
    """
    Bilinearly interpolate dense features at keypoint locations
    Args:
        keypoints: [B, N, 2] keypoint locations in pixel coordinates
        dense: [B, C, H, W] dense feature map
    Returns:
        features: [B, N, C] interpolated features
    """
    b, c, h, w = dense.shape

    # Normalize keypoints to [-1, 1] for grid_sample
    keypoints_norm = keypoints.clone()
    keypoints_norm[:, :, 0] = 2.0 * keypoints[:, :, 0] / (w - 1) - 1.0
    keypoints_norm[:, :, 1] = 2.0 * keypoints[:, :, 1] / (h - 1) - 1.0

    # Reshape for grid_sample: [B, N, 2] -> [B, 1, N, 2]
    keypoints_norm = keypoints_norm.unsqueeze(1)

    # Interpolate
    features = F.grid_sample(
        dense,
        keypoints_norm,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros'
    )

    # Reshape: [B, C, 1, N] -> [B, N, C]
    features = features.squeeze(2).permute(0, 2, 1)

    return features


def interpolate_sparse_to_dense(keypoints, values, output_shape):
    """
    Interpolate sparse values at keypoints to a dense grid
    Args:
        keypoints: [B, N, 2] keypoint locations
        values: [B, N, C] values at keypoints
        output_shape: (H, W) output shape
    Returns:
        dense: [B, C, H, W] dense interpolated values
    """
    b, n, c = values.shape
    h, w = output_shape

    # Create output grid
    device = keypoints.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )

    grid_points = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
    grid_points = grid_points.reshape(1, -1, 2).expand(b, -1, -1)  # [B, H*W, 2]

    # Compute distances from each grid point to each keypoint
    # [B, H*W, 1, 2] - [B, 1, N, 2] -> [B, H*W, N, 2]
    distances = grid_points.unsqueeze(2) - keypoints.unsqueeze(1)
    distances = torch.norm(distances, dim=-1)  # [B, H*W, N]

    # Use inverse distance weighting
    epsilon = 1e-8
    weights = 1.0 / (distances + epsilon)  # [B, H*W, N]
    weights = weights / (weights.sum(dim=-1, keepdim=True) + epsilon)  # Normalize

    # Weighted sum: [B, H*W, N] @ [B, N, C] -> [B, H*W, C]
    dense_flat = torch.bmm(weights, values)

    # Reshape to image format
    dense = dense_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]

    return dense
