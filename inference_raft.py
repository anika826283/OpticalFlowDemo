"""
RAFT Inference Script
This script performs optical flow estimation on a pair of images using RAFT
"""
import argparse
import os
import torch
import numpy as np
from pathlib import Path

from raft import RAFT
from utils import (
    load_image,
    prepare_image_tensor,
    save_flow_visualization,
    save_flow_numpy,
    pad_image,
    unpad_flow
)


def load_model(checkpoint_path=None, small=False, device='cuda'):
    """
    Load RAFT model
    Args:
        checkpoint_path: path to checkpoint file
        small: use small model for faster inference
        device: device to load model on
    Returns:
        model: RAFT model
    """
    print("Initializing RAFT model...")
    if small:
        print("Using RAFT-small for faster inference")

    model = RAFT(small=small, dropout=0.0, alternate_corr=False, mixed_precision=False)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint provided or file not found. Using randomly initialized weights.")
        print("For best results, download pretrained weights from:")
        print("https://github.com/princeton-vl/RAFT")

    model.to(device)
    model.eval()

    return model


def inference_on_image_pair(model, img0_path, img1_path, output_dir, iters=12, device='cuda'):
    """
    Run inference on a pair of images
    Args:
        model: RAFT model
        img0_path: path to first image
        img1_path: path to second image
        output_dir: directory to save output
        iters: number of refinement iterations
        device: device to run inference on
    """
    print(f"\nProcessing image pair:")
    print(f"  Image 0: {img0_path}")
    print(f"  Image 1: {img1_path}")

    # Load images
    img0 = load_image(img0_path)
    img1 = load_image(img1_path)

    print(f"Image shape: {img0.shape}")

    # Prepare tensors
    img0_tensor = prepare_image_tensor(img0).to(device)
    img1_tensor = prepare_image_tensor(img1).to(device)

    # Note: RAFT expects images in [0, 255] range
    img0_tensor = img0_tensor * 255.0
    img1_tensor = img1_tensor * 255.0

    # Pad images to be divisible by 8
    img0_padded, pad = pad_image(img0_tensor, divisor=8)
    img1_padded, _ = pad_image(img1_tensor, divisor=8)

    # Run inference
    print(f"Running optical flow estimation with {iters} iterations...")
    with torch.no_grad():
        flow = model.inference(img0_padded, img1_padded, iters=iters)

    # Remove padding
    flow = unpad_flow(flow, pad)

    # Convert to numpy
    flow_np = flow[0].cpu().numpy()  # [2, H, W]

    print(f"Flow shape: {flow_np.shape}")
    print(f"Flow magnitude range: [{np.min(np.sqrt(flow_np[0]**2 + flow_np[1]**2)):.2f}, "
          f"{np.max(np.sqrt(flow_np[0]**2 + flow_np[1]**2)):.2f}]")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames
    img0_name = Path(img0_path).stem
    img1_name = Path(img1_path).stem
    output_prefix = f"{img0_name}_to_{img1_name}_raft"

    # Save flow visualization
    vis_path = os.path.join(output_dir, f"{output_prefix}_flow_vis.png")
    save_flow_visualization(flow_np, vis_path)

    # Save flow as numpy array
    npy_path = os.path.join(output_dir, f"{output_prefix}_flow.npy")
    save_flow_numpy(flow_np, npy_path)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='RAFT Optical Flow Inference')

    parser.add_argument('--img0', type=str, required=True,
                        help='Path to first image')
    parser.add_argument('--img1', type=str, required=True,
                        help='Path to second image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='data/output',
                        help='Directory to save output (default: data/output)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--small', action='store_true',
                        help='Use RAFT-small for faster inference')
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of refinement iterations (default: 12)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'

    # Load model
    model = load_model(args.checkpoint, args.small, args.device)

    # Run inference
    inference_on_image_pair(
        model,
        args.img0,
        args.img1,
        args.output_dir,
        args.iters,
        args.device
    )

    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
