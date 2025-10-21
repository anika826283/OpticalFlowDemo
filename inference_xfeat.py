"""
XFeat Inference Script
This script performs feature matching and sparse optical flow estimation using XFeat
"""
import argparse
import os
import torch
import numpy as np
import cv2
from pathlib import Path

from xfeat import XFeat
from utils import load_image, prepare_image_tensor, pad_image, unpad_flow


def load_model(checkpoint_path=None, device='cuda'):
    """
    Load XFeat model
    Args:
        checkpoint_path: path to checkpoint file
        device: device to load model on
    Returns:
        model: XFeat model
    """
    print("Initializing XFeat model...")

    model = XFeat(
        feature_dim=64,
        max_keypoints=4096,
        detection_threshold=0.005
    )

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
        print("https://github.com/verlab/accelerated_features")

    model.to(device)
    model.eval()

    return model


def visualize_matches(img0, img1, kpts0, kpts1, matches, output_path):
    """
    Visualize feature matches
    Args:
        img0: first image [H, W, 3]
        img1: second image [H, W, 3]
        kpts0: keypoints in image0 [N, 2]
        kpts1: keypoints in image1 [N, 2]
        matches: match indices [M, 2]
        output_path: path to save visualization
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Create side-by-side image
    h = max(h0, h1)
    w_total = w0 + w1

    # Resize images to same height if needed
    if h0 != h:
        img0 = cv2.resize(img0, (int(w0 * h / h0), h))
    if h1 != h:
        img1 = cv2.resize(img1, (int(w1 * h / h1), h))

    vis_img = np.zeros((h, w_total, 3), dtype=np.uint8)
    vis_img[:h0, :w0] = img0
    vis_img[:h1, w0:w0+w1] = img1

    # Draw matches
    if len(matches) > 0:
        for i in range(min(len(matches), 500)):  # Limit to 500 matches for visualization
            idx0, idx1 = matches[i]
            pt0 = (int(kpts0[idx0, 0]), int(kpts0[idx0, 1]))
            pt1 = (int(kpts1[idx1, 0] + w0), int(kpts1[idx1, 1]))

            # Random color for each match
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Draw line
            cv2.line(vis_img, pt0, pt1, color, 1)
            # Draw circles
            cv2.circle(vis_img, pt0, 2, color, -1)
            cv2.circle(vis_img, pt1, 2, color, -1)

    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Match visualization saved to {output_path}")


def visualize_sparse_flow(img0, sparse_flow_dict, output_path):
    """
    Visualize sparse optical flow
    Args:
        img0: first image [H, W, 3]
        sparse_flow_dict: dict containing keypoints, flow, and confidence
        output_path: path to save visualization
    """
    vis_img = img0.copy()

    keypoints = sparse_flow_dict['keypoints']
    flow = sparse_flow_dict['flow']

    if len(keypoints) > 0:
        for i in range(len(keypoints)):
            pt0 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt1 = (int(keypoints[i, 0] + flow[i, 0]), int(keypoints[i, 1] + flow[i, 1]))

            # Color based on flow magnitude
            magnitude = np.sqrt(flow[i, 0]**2 + flow[i, 1]**2)
            color = plt_colormap(magnitude / 50.0)  # Normalize to [0, 1]
            color = tuple([int(c * 255) for c in color[:3]])

            # Draw arrow
            cv2.arrowedLine(vis_img, pt0, pt1, color, 2, tipLength=0.3)

    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"Sparse flow visualization saved to {output_path}")


def plt_colormap(value):
    """Simple colormap for visualization"""
    # Simple red-to-blue colormap
    r = max(0, min(1, 2 * value))
    b = max(0, min(1, 2 * (1 - value)))
    g = 1 - abs(2 * value - 1)
    return (r, g, b, 1.0)


def inference_on_image_pair(model, img0_path, img1_path, output_dir, device='cuda'):
    """
    Run inference on a pair of images
    Args:
        model: XFeat model
        img0_path: path to first image
        img1_path: path to second image
        output_dir: directory to save output
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

    # XFeat expects images in [0, 255] range
    img0_tensor = img0_tensor * 255.0
    img1_tensor = img1_tensor * 255.0

    # Pad images to be divisible by 8
    img0_padded, pad = pad_image(img0_tensor, divisor=8)
    img1_padded, _ = pad_image(img1_tensor, divisor=8)

    # Run inference
    print("Running feature extraction and matching...")
    with torch.no_grad():
        results = model.inference(img0_padded, img1_padded)

    # Compute sparse flow
    sparse_flow = model.compute_sparse_flow(results, img0.shape[:2])

    # Get results for first batch
    kpts0 = results['keypoints0'][0].cpu().numpy()
    kpts1 = results['keypoints1'][0].cpu().numpy()
    matches = results['matches'][0].cpu().numpy()
    sparse_flow_dict = {
        'keypoints': sparse_flow[0]['keypoints'].cpu().numpy(),
        'flow': sparse_flow[0]['flow'].cpu().numpy(),
        'confidence': sparse_flow[0]['confidence'].cpu().numpy()
    }

    print(f"Detected keypoints: {len(kpts0)} in image0, {len(kpts1)} in image1")
    print(f"Number of matches: {len(matches)}")
    print(f"Sparse flow vectors: {len(sparse_flow_dict['flow'])}")

    if len(sparse_flow_dict['flow']) > 0:
        flow_magnitudes = np.sqrt(sparse_flow_dict['flow'][:, 0]**2 + sparse_flow_dict['flow'][:, 1]**2)
        print(f"Flow magnitude: mean={np.mean(flow_magnitudes):.2f}, max={np.max(flow_magnitudes):.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames
    img0_name = Path(img0_path).stem
    img1_name = Path(img1_path).stem
    output_prefix = f"{img0_name}_to_{img1_name}_xfeat"

    # Save match visualization
    matches_path = os.path.join(output_dir, f"{output_prefix}_matches.png")
    visualize_matches(img0, img1, kpts0, kpts1, matches, matches_path)

    # Save sparse flow visualization
    flow_path = os.path.join(output_dir, f"{output_prefix}_sparse_flow.png")
    visualize_sparse_flow(img0, sparse_flow_dict, flow_path)

    # Save data as numpy arrays
    data_path = os.path.join(output_dir, f"{output_prefix}_data.npz")
    np.savez(data_path,
             keypoints0=kpts0,
             keypoints1=kpts1,
             matches=matches,
             sparse_keypoints=sparse_flow_dict['keypoints'],
             sparse_flow=sparse_flow_dict['flow'],
             confidence=sparse_flow_dict['confidence'])
    print(f"Data saved to {data_path}")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='XFeat Feature Matching and Sparse Flow Inference')

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

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Run inference
    inference_on_image_pair(
        model,
        args.img0,
        args.img1,
        args.output_dir,
        args.device
    )

    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
