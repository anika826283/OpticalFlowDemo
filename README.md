# OpticalFlowDemo

Optical Flow Estimation with GMFlow and RAFT

## Introduction

This project implements the inference functionality of two state-of-the-art optical flow estimation methods:

- **GMFlow** (Global Matching Flow) - A global matching-based method presented at CVPR 2022
- **RAFT** (Recurrent All-Pairs Field Transforms) - An iterative refinement-based method presented at ECCV 2020

Both models are implemented for estimating optical flow between consecutive images with high accuracy and efficiency.

## Project Structure

```
OpticalFlowDemo/
├── gmflow/                    # GMFlow model implementation
│   ├── __init__.py
│   ├── gmflow.py             # Main model
│   ├── backbone.py           # CNN feature extraction network
│   ├── transformer.py        # Transformer module
│   └── geometry.py           # Geometry utility functions
├── raft/                      # RAFT model implementation
│   ├── __init__.py
│   ├── raft.py               # Main model
│   ├── extractor.py          # Feature extraction network
│   ├── corr.py               # Correlation volume
│   ├── update.py             # GRU-based update operator
│   └── utils.py              # RAFT utility functions
├── data/
│   ├── input/                # Input images directory
│   └── output/               # Output results directory
├── checkpoints/              # Model weights directory
├── inference.py              # GMFlow inference script
├── inference_raft.py         # RAFT inference script
├── example.py                # GMFlow example code
├── example_raft.py           # RAFT example code
├── utils.py                  # Shared utility functions
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

**GMFlow:**
1. Visit [GMFlow GitHub](https://github.com/haofeixu/gmflow)
2. Download pretrained model (e.g., `gmflow_sintel-0c07dcb3.pth`)
3. Place the weight file in the `checkpoints/` directory

```bash
# Example (adjust download link as needed)
cd checkpoints
wget https://github.com/haofeixu/gmflow/releases/download/v0.1/gmflow_sintel-0c07dcb3.pth
cd ..
```

**RAFT:**
1. Visit [RAFT GitHub](https://github.com/princeton-vl/RAFT)
2. Download pretrained model (e.g., `raft-things.pth`, `raft-sintel.pth`)
3. Place the weight file in the `checkpoints/` directory

```bash
# Example (adjust download link as needed)
cd checkpoints
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth
cd ..
```

## Usage

### GMFlow

#### Method 1: Command Line Script

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

#### Method 2: Run Example Script

Run the example script to test the model (uses synthetic images):

```bash
python example.py
```

### RAFT

#### Method 1: Command Line Script

```bash
python inference_raft.py \
    --img0 data/input/frame1.png \
    --img1 data/input/frame2.png \
    --checkpoint checkpoints/raft-things.pth \
    --output_dir data/output \
    --device cuda \
    --iters 12
```

Parameter descriptions:
- `--img0`: Path to first image (required)
- `--img1`: Path to second image (required)
- `--checkpoint`: Path to model checkpoint file (optional)
- `--output_dir`: Output directory (default: `data/output`)
- `--device`: Computing device, `cuda` or `cpu` (default: `cuda`)
- `--small`: Use RAFT-small for faster inference (optional flag)
- `--iters`: Number of refinement iterations (default: 12)

#### Method 2: Run Example Script

Run the example script to test RAFT (uses synthetic images):

```bash
python example_raft.py
```

### Method 3: Use in Code

**GMFlow:**
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

**RAFT:**
```python
import torch
from raft import RAFT
from utils import load_image, prepare_image_tensor, visualize_flow, pad_image, unpad_flow

# Initialize model
model = RAFT(small=False, dropout=0.0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Load images
img0 = load_image('path/to/img0.jpg')
img1 = load_image('path/to/img1.jpg')

# Convert to tensors (RAFT expects [0, 255] range)
img0_tensor = prepare_image_tensor(img0).to(device) * 255.0
img1_tensor = prepare_image_tensor(img1).to(device) * 255.0

# Padding
img0_padded, pad = pad_image(img0_tensor, divisor=8)
img1_padded, _ = pad_image(img1_tensor, divisor=8)

# Run inference with iterative refinement
with torch.no_grad():
    flow = model.inference(img0_padded, img1_padded, iters=12)

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

### GMFlow

- **GPU Mode** (recommended):
  - Input resolution 640×480: ~50-100ms (depending on GPU model)
  - Requires ~2-4GB VRAM

- **CPU Mode**:
  - Input resolution 640×480: ~2-5 seconds
  - Requires ~4GB RAM

### RAFT

- **GPU Mode** (recommended):
  - Input resolution 640×480: ~80-150ms with 12 iterations (depending on GPU model)
  - RAFT-small: ~40-80ms with 12 iterations
  - Requires ~3-5GB VRAM

- **CPU Mode**:
  - Input resolution 640×480: ~5-10 seconds with 12 iterations
  - Requires ~4-6GB RAM

**Note:** RAFT's performance scales with the number of iterations. Fewer iterations (e.g., 6) will be faster but less accurate.

## Citation

If you use this project, please cite the original papers:

**GMFlow:**
```bibtex
@inproceedings{xu2022gmflow,
  title={GMFlow: Learning Optical Flow via Global Matching},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8121-8130},
  year={2022}
}
```

**RAFT:**
```bibtex
@inproceedings{teed2020raft,
  title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={402-419},
  year={2020}
}
```

## Related Resources

### GMFlow
- [GMFlow Official GitHub](https://github.com/haofeixu/gmflow)
- [GMFlow Paper](https://arxiv.org/abs/2111.13680)
- [CVPR 2022 Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)

### RAFT
- [RAFT Official GitHub](https://github.com/princeton-vl/RAFT)
- [RAFT Paper](https://arxiv.org/abs/2003.12039)
- [ECCV 2020 Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf)

## License

This project is for learning and research purposes only. Please refer to the official repositories for the original model licenses.

## FAQ

### Q: Which model should I use, GMFlow or RAFT?

A:
- **GMFlow**: Generally faster, good for real-time applications, single-pass inference
- **RAFT**: More accurate on challenging scenes, iterative refinement allows trading speed for accuracy
- **Recommendation**: Start with GMFlow for speed, use RAFT if you need higher accuracy

### Q: Can I run this without a GPU?

A: Yes! Use the `--device cpu` parameter to run on CPU, though it will be slower.

### Q: What image formats are supported?

A: Common image formats including PNG, JPG, JPEG, BMP, etc.

### Q: How to improve execution speed?

A:
1. Use GPU (CUDA)
2. For GMFlow: Already optimized for speed
3. For RAFT: Use `--small` flag or reduce `--iters` (e.g., 6 instead of 12)
4. Reduce input image resolution

### Q: What if I get a CUDA out of memory error?

A:
1. Reduce input image resolution
2. For RAFT: Use `--small` flag for RAFT-small model
3. Use CPU mode
4. Close other programs using GPU

## Contact

For questions or suggestions, please open an Issue for discussion.
