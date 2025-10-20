"""
Utility functions for optical flow visualization and I/O
"""
import numpy as np
import cv2
import torch
from PIL import Image


def load_image(image_path):
    """
    Load image from path
    Args:
        image_path: path to image
    Returns:
        image: numpy array [H, W, 3] in RGB
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    return image


def prepare_image_tensor(image):
    """
    Convert image to tensor
    Args:
        image: numpy array [H, W, 3] in RGB, value range [0, 255]
    Returns:
        tensor: [1, 3, H, W] in range [0, 1]
    """
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)
    image = image / 255.0
    return image


def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to color image for visualization
    Args:
        flow: [H, W, 2] numpy array
        max_flow: maximum flow magnitude for normalization
    Returns:
        color_image: [H, W, 3] numpy array in RGB
    """
    h, w = flow.shape[:2]

    # Convert to polar coordinates
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    # Compute angle and magnitude
    rad = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)

    # Normalize
    if max_flow is None:
        max_flow = np.max(rad)

    if max_flow > 0:
        rad = rad / max_flow

    # Convert to HSV
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue
    hsv[:, :, 1] = 255  # Saturation
    hsv[:, :, 2] = (np.minimum(rad, 1.0) * 255).astype(np.uint8)  # Value

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def visualize_flow(flow, max_flow=None):
    """
    Visualize optical flow
    Args:
        flow: [2, H, W] or [H, W, 2] tensor or numpy array
        max_flow: maximum flow magnitude for normalization
    Returns:
        vis_image: [H, W, 3] numpy array
    """
    # Convert to numpy if tensor
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    # Convert [2, H, W] to [H, W, 2]
    if flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)

    # Convert to color
    vis_image = flow_to_color(flow, max_flow)

    return vis_image


def save_flow_visualization(flow, save_path, max_flow=None):
    """
    Save flow visualization to file
    Args:
        flow: [2, H, W] or [H, W, 2] tensor or numpy array
        save_path: path to save visualization
        max_flow: maximum flow magnitude for normalization
    """
    vis_image = visualize_flow(flow, max_flow)
    Image.fromarray(vis_image).save(save_path)
    print(f"Flow visualization saved to {save_path}")


def save_flow_numpy(flow, save_path):
    """
    Save flow as numpy array
    Args:
        flow: [2, H, W] or [H, W, 2] tensor or numpy array
        save_path: path to save flow (.npy)
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    np.save(save_path, flow)
    print(f"Flow array saved to {save_path}")


def create_color_wheel():
    """
    Create color wheel for flow visualization
    Returns:
        color_wheel: [55, 3] numpy array
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    color_wheel = np.zeros((ncols, 3))

    col = 0
    # RY
    color_wheel[col:col+RY, 0] = 255
    color_wheel[col:col+RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY

    # YG
    color_wheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    color_wheel[col:col+YG, 1] = 255
    col += YG

    # GC
    color_wheel[col:col+GC, 1] = 255
    color_wheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC

    # CB
    color_wheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    color_wheel[col:col+CB, 2] = 255
    col += CB

    # BM
    color_wheel[col:col+BM, 2] = 255
    color_wheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM

    # MR
    color_wheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    color_wheel[col:col+MR, 0] = 255

    return color_wheel


def pad_image(img, divisor=8):
    """
    Pad image to be divisible by divisor
    Args:
        img: [B, C, H, W] tensor
        divisor: divisor for padding
    Returns:
        padded_img: [B, C, H', W'] tensor
        (pad_h, pad_w): padding sizes
    """
    b, c, h, w = img.shape

    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='replicate')

    return img, (pad_h, pad_w)


def unpad_flow(flow, pad):
    """
    Remove padding from flow
    Args:
        flow: [B, 2, H, W] tensor
        pad: (pad_h, pad_w)
    Returns:
        unpadded_flow: [B, 2, H', W'] tensor
    """
    pad_h, pad_w = pad

    if pad_h > 0:
        flow = flow[:, :, :-pad_h, :]
    if pad_w > 0:
        flow = flow[:, :, :, :-pad_w]

    return flow
