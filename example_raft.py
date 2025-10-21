"""
Example script for RAFT optical flow inference
This demonstrates how to use RAFT programmatically
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from raft import RAFT
from utils import load_image, prepare_image_tensor, visualize_flow, pad_image, unpad_flow


def run_example():
    """
    Example of running RAFT inference programmatically
    """
    # Initialize model
    print("Initializing RAFT model...")
    model = RAFT(
        small=False,  # Set to True for faster inference with RAFT-small
        dropout=0.0,
        alternate_corr=False,
        mixed_precision=False
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

    # Simulate motion by shifting img0
    shift_x, shift_y = 10, 5
    img1 = np.zeros_like(img0)
    img1[shift_y:, shift_x:] = img0[:-shift_y, :-shift_x]

    # Convert to tensors
    img0_tensor = prepare_image_tensor(img0).to(device)
    img1_tensor = prepare_image_tensor(img1).to(device)

    # RAFT expects images in [0, 255] range
    img0_tensor = img0_tensor * 255.0
    img1_tensor = img1_tensor * 255.0

    # Pad images
    img0_padded, pad = pad_image(img0_tensor, divisor=8)
    img1_padded, _ = pad_image(img1_tensor, divisor=8)

    # Run inference
    print("Running optical flow estimation with RAFT...")
    iters = 12  # Number of refinement iterations
    with torch.no_grad():
        flow = model.inference(img0_padded, img1_padded, iters=iters)

    # Remove padding
    flow = unpad_flow(flow, pad)

    # Visualize results
    print("Visualizing results...")
    flow_vis = visualize_flow(flow[0])

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img0)
    axes[0].set_title('Image 0')
    axes[0].axis('off')

    axes[1].imshow(img1)
    axes[1].set_title('Image 1')
    axes[1].axis('off')

    axes[2].imshow(flow_vis)
    axes[2].set_title('RAFT Optical Flow Visualization')
    axes[2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = 'data/output/example_result_raft.png'
    Path('data/output').mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResult saved to {output_path}")

    # Print flow statistics
    flow_np = flow[0].cpu().numpy()
    flow_magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)

    print(f"\nFlow Statistics:")
    print(f"  Mean magnitude: {np.mean(flow_magnitude):.2f} pixels")
    print(f"  Max magnitude: {np.max(flow_magnitude):.2f} pixels")
    print(f"  Mean flow X: {np.mean(flow_np[0]):.2f} pixels")
    print(f"  Mean flow Y: {np.mean(flow_np[1]):.2f} pixels")

    print("\nExample completed successfully!")
    print("\nTo use with real images, run:")
    print("  python inference_raft.py --img0 path/to/img0.jpg --img1 path/to/img1.jpg")


if __name__ == '__main__':
    run_example()
