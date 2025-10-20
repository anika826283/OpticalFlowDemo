# OpticalFlowDemo

GMFlow-based Optical Flow Estimation Demo Project

## Introduction

This project implements the inference functionality of GMFlow (Global Matching Flow) for estimating optical flow between consecutive images. GMFlow is a deep learning-based optical flow estimation method based on global matching, presented at CVPR 2022.

## Project Structure

```
OpticalFlowDemo/
├── gmflow/                    # GMFlow model implementation
│   ├── __init__.py
│   ├── gmflow.py             # Main model
│   ├── backbone.py           # CNN feature extraction network
│   ├── transformer.py        # Transformer module
│   └── geometry.py           # Geometry utility functions
├── data/
│   ├── input/                # Input images directory
│   └── output/               # Output results directory
├── checkpoints/              # Model weights directory
├── inference.py              # Main inference script
├── example.py                # Example code
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Environment Setup

### System Requirements

- Python 3.8 or higher
- CUDA 11.0 or higher (if using GPU)
- 4GB+ RAM (CPU mode) or 4GB+ VRAM (GPU mode)

### Step 1: Install Python Environment

We recommend using a virtual environment to manage dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - PyTorch vision utilities
- `numpy>=1.21.0` - Numerical computing
- `opencv-python>=4.5.0` - Image processing
- `pillow>=9.0.0` - Image I/O
- `matplotlib>=3.5.0` - Visualization
- `einops>=0.6.0` - Tensor operations
- `timm>=0.9.0` - Vision model utilities

### Step 3: (Optional) Download Pretrained Weights

For best results, we recommend downloading the official pretrained weights:

1. Visit [GMFlow GitHub](https://github.com/haofeixu/gmflow)
2. Download pretrained model (e.g., `gmflow_sintel-0c07dcb3.pth`)
3. Place the weight file in the `checkpoints/` directory

```bash
# Example (adjust download link as needed)
cd checkpoints
wget https://github.com/haofeixu/gmflow/releases/download/v0.1/gmflow_sintel-0c07dcb3.pth
cd ..
```

## Usage

### Method 1: Command Line Script

```bash
python inference.py \
    --img0 data/input/frame1.png \
    --img1 data/input/frame2.png \
    --checkpoint checkpoints/gmflow_sintel-0c07dcb3.pth \
    --output_dir data/output \
    --device cuda
```

Parameter descriptions:
- `--img0`: Path to first image (required)
- `--img1`: Path to second image (required)
- `--checkpoint`: Path to model checkpoint file (optional, uses random initialization if not provided)
- `--output_dir`: Output directory (default: `data/output`)
- `--device`: Computing device, `cuda` or `cpu` (default: `cuda`)

### Method 2: Run Example Script

Run the example script to test the model (uses synthetic images):

```bash
python example.py
```

### Method 3: Use in Code

```python
import torch
from gmflow import GMFlow
from utils import load_image, prepare_image_tensor, visualize_flow, pad_image, unpad_flow

# Initialize model
model = GMFlow(
    num_scales=1,
    feature_channels=128,
    upsample_factor=8,
    num_transformer_layers=6,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Load images
img0 = load_image('path/to/img0.jpg')
img1 = load_image('path/to/img1.jpg')

# Convert to tensors
img0_tensor = prepare_image_tensor(img0).to(device)
img1_tensor = prepare_image_tensor(img1).to(device)

# Padding
img0_padded, pad = pad_image(img0_tensor, divisor=8)
img1_padded, _ = pad_image(img1_tensor, divisor=8)

# Run inference
with torch.no_grad():
    flow = model.inference(img0_padded, img1_padded)

# Remove padding
flow = unpad_flow(flow, pad)

# Visualize
flow_vis = visualize_flow(flow[0])
```

## Output Description

The inference script generates two output files:

1. `*_flow_vis.png` - Flow visualization
   - Uses color encoding to represent flow direction and magnitude
   - Hue represents direction
   - Value represents flow intensity

2. `*_flow.npy` - Raw flow data
   - NumPy array format
   - Shape: `[2, H, W]`, where `[0]` is x-direction flow, `[1]` is y-direction flow

## Performance

- **GPU Mode** (recommended):
  - Input resolution 640×480: ~50-100ms (depending on GPU model)
  - Requires ~2-4GB VRAM

- **CPU Mode**:
  - Input resolution 640×480: ~2-5 seconds
  - Requires ~4GB RAM

## Citation

If you use this project, please cite the original GMFlow paper:

```bibtex
@inproceedings{xu2022gmflow,
  title={GMFlow: Learning Optical Flow via Global Matching},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8121-8130},
  year={2022}
}
```

## Related Resources

- [GMFlow Official GitHub](https://github.com/haofeixu/gmflow)
- [GMFlow Paper](https://arxiv.org/abs/2111.13680)
- [CVPR 2022 Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)

## License

This project is for learning and research purposes only. Please refer to the official repository for the original GMFlow model license.

## FAQ

### Q: Can I run this without a GPU?

A: Yes! Use the `--device cpu` parameter to run on CPU, though it will be slower.

### Q: What image formats are supported?

A: Common image formats including PNG, JPG, JPEG, BMP, etc.

### Q: How to improve execution speed?

A:
1. Use GPU (CUDA)
2. Reduce input image resolution
3. Use pretrained weights (may affect accuracy)

### Q: What if I get a CUDA out of memory error?

A:
1. Reduce input image resolution
2. Use CPU mode
3. Close other programs using GPU

## Contact

For questions or suggestions, please open an Issue for discussion.
