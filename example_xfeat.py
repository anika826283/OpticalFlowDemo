"""
Example script for XFeat feature matching and sparse flow
This demonstrates how to use XFeat programmatically
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from xfeat import XFeat
from utils import load_image, prepare_image_tensor, pad_image


def run_example():
    """
    Example of running XFeat inference programmatically
    """
    # Initialize model
    print("Initializing XFeat model...")
    model = XFeat(
        feature_dim=64,
        max_keypoints=4096,
        detection_threshold=0.005
    )

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    # Example: Create synthetic image pair with known motion
    print("\nCreating synthetic image pair for testing...")

    # Create a simple pattern (you can replace this with real images)
    h, w = 480, 640
    img0 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Add some distinctive features
    for i in range(20):
        x, y = np.random.randint(50, w-50), np.random.randint(50, h-50)
        size = np.random.randint(10, 30)
        color = tuple(np.random.randint(100, 255, 3).tolist())
        img0 = np.ascontiguousarray(img0)
        # Draw rectangles as features
        img0[y:y+size, x:x+size] = color

    # Simulate motion by shifting img0
    shift_x, shift_y = 20, 10
    img1 = np.zeros_like(img0)
    img1[shift_y:, shift_x:] = img0[:-shift_y, :-shift_x]

    # Convert to tensors
    img0_tensor = prepare_image_tensor(img0).to(device)
    img1_tensor = prepare_image_tensor(img1).to(device)

    # XFeat expects images in [0, 255] range
    img0_tensor = img0_tensor * 255.0
    img1_tensor = img1_tensor * 255.0

    # Pad images
    img0_padded, pad = pad_image(img0_tensor, divisor=8)
    img1_padded, _ = pad_image(img1_tensor, divisor=8)

    # Run inference
    print("Running feature extraction and matching with XFeat...")
    with torch.no_grad():
        results = model.inference(img0_padded, img1_padded)

    # Compute sparse flow
    sparse_flow = model.compute_sparse_flow(results, (h, w))

    # Get results
    kpts0 = results['keypoints0'][0].cpu().numpy()
    kpts1 = results['keypoints1'][0].cpu().numpy()
    matches = results['matches'][0].cpu().numpy()
    flow_data = sparse_flow[0]

    print(f"\nFeature Extraction Results:")
    print(f"  Keypoints in image 0: {len(kpts0)}")
    print(f"  Keypoints in image 1: {len(kpts1)}")
    print(f"  Number of matches: {len(matches)}")

    if len(matches) > 0:
        # Get matched keypoints
        flow_vectors = flow_data['flow'].cpu().numpy()
        flow_magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)

        print(f"\nSparse Flow Statistics:")
        print(f"  Number of flow vectors: {len(flow_vectors)}")
        print(f"  Mean flow magnitude: {np.mean(flow_magnitudes):.2f} pixels")
        print(f"  Max flow magnitude: {np.max(flow_magnitudes):.2f} pixels")
        print(f"  Mean flow X: {np.mean(flow_vectors[:, 0]):.2f} pixels")
        print(f"  Mean flow Y: {np.mean(flow_vectors[:, 1]):.2f} pixels")

        # Visualize results
        print("\nVisualizing results...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Image 0 with keypoints
        axes[0].imshow(img0)
        if len(kpts0) > 0:
            axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='red', s=10, alpha=0.5)
        axes[0].set_title(f'Image 0 ({len(kpts0)} keypoints)')
        axes[0].axis('off')

        # Image 1 with keypoints
        axes[1].imshow(img1)
        if len(kpts1) > 0:
            axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='red', s=10, alpha=0.5)
        axes[1].set_title(f'Image 1 ({len(kpts1)} keypoints)')
        axes[1].axis('off')

        # Sparse flow visualization
        axes[2].imshow(img0)
        matched_kpts0 = kpts0[matches[:, 0]]
        for i in range(min(len(flow_vectors), 200)):  # Limit to 200 arrows for clarity
            x0, y0 = matched_kpts0[i]
            dx, dy = flow_vectors[i]
            axes[2].arrow(x0, y0, dx, dy, head_width=3, head_length=3,
                         fc='yellow', ec='red', alpha=0.7, linewidth=1)
        axes[2].set_title(f'Sparse Flow ({len(flow_vectors)} vectors)')
        axes[2].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = 'data/output/example_result_xfeat.png'
        Path('data/output').mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nResult saved to {output_path}")

        print("\nExample completed successfully!")
    else:
        print("\nNo matches found. This is expected with random noise images.")
        print("Try using real images with distinctive features for better results.")

    print("\nTo use with real images, run:")
    print("  python inference_xfeat.py --img0 path/to/img0.jpg --img1 path/to/img1.jpg")


if __name__ == '__main__':
    run_example()
