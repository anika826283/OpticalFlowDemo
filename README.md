# OpticalFlowDemo

Optical Flow Estimation with GMFlow, RAFT, and XFeat

## Introduction

This project implements the inference functionality of three state-of-the-art optical flow and feature matching methods:

- **GMFlow** (Global Matching Flow) - Dense optical flow via global matching, presented at CVPR 2022
- **RAFT** (Recurrent All-Pairs Field Transforms) - Dense optical flow via iterative refinement, presented at ECCV 2020
- **XFeat** (Accelerated Features) - Lightweight feature matching for sparse optical flow, presented at CVPR 2024

These models provide different approaches to motion estimation:
- **Dense flow** (GMFlow, RAFT): Estimates motion for every pixel
- **Sparse flow** (XFeat): Estimates motion at detected feature points, extremely fast

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
├── xfeat/                     # XFeat model implementation
│   ├── __init__.py
│   ├── xfeat.py              # Main model
│   ├── net.py                # Neural network components
│   └── interpolator.py       # Interpolation utilities
├── data/
│   ├── input/                # Input images directory
│   └── output/               # Output results directory
├── checkpoints/              # Model weights directory
├── inference.py              # GMFlow inference script
├── inference_raft.py         # RAFT inference script
├── inference_xfeat.py        # XFeat inference script
├── example.py                # GMFlow example code
├── example_raft.py           # RAFT example code
├── example_xfeat.py          # XFeat example code
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

**XFeat:**
1. Visit [XFeat GitHub](https://github.com/verlab/accelerated_features)
2. Download pretrained model (e.g., `xfeat.pth`)
3. Place the weight file in the `checkpoints/` directory

```bash
# Example (adjust download link as needed)
cd checkpoints
wget https://github.com/verlab/accelerated_features/releases/download/v1.0/xfeat.pth
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

### XFeat

#### Method 1: Command Line Script

```bash
python inference_xfeat.py \
    --img0 data/input/frame1.png \
    --img1 data/input/frame2.png \
    --checkpoint checkpoints/xfeat.pth \
    --output_dir data/output \
    --device cuda
```

Parameter descriptions:
- `--img0`: Path to first image (required)
- `--img1`: Path to second image (required)
- `--checkpoint`: Path to model checkpoint file (optional)
- `--output_dir`: Output directory (default: `data/output`)
- `--device`: Computing device, `cuda` or `cpu` (default: `cuda`)

#### Method 2: Run Example Script

Run the example script to test XFeat (uses synthetic images):

```bash
python example_xfeat.py
```

#### Method 3: Use in Code

```python
import torch
from xfeat import XFeat
from utils import load_image, prepare_image_tensor, pad_image

# Initialize model
model = XFeat(feature_dim=64, max_keypoints=4096, detection_threshold=0.005)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Load images
img0 = load_image('path/to/img0.jpg')
img1 = load_image('path/to/img1.jpg')

# Convert to tensors (XFeat expects [0, 255] range)
img0_tensor = prepare_image_tensor(img0).to(device) * 255.0
img1_tensor = prepare_image_tensor(img1).to(device) * 255.0

# Padding
img0_padded, pad = pad_image(img0_tensor, divisor=8)
img1_padded, _ = pad_image(img1_tensor, divisor=8)

# Run inference
with torch.no_grad():
    results = model.inference(img0_padded, img1_padded)
    sparse_flow = model.compute_sparse_flow(results, img0.shape[:2])

# Access results
keypoints0 = results['keypoints0'][0]  # Detected keypoints in image 0
keypoints1 = results['keypoints1'][0]  # Detected keypoints in image 1
matches = results['matches'][0]        # Match indices
flow_vectors = sparse_flow[0]['flow']  # Sparse flow vectors
```

## Output Description

### Dense Flow Models (GMFlow, RAFT)

The inference scripts generate two output files:

1. `*_flow_vis.png` - Flow visualization
   - Uses color encoding to represent flow direction and magnitude
   - Hue represents direction
   - Value represents flow intensity

2. `*_flow.npy` - Raw flow data
   - NumPy array format
   - Shape: `[2, H, W]`, where `[0]` is x-direction flow, `[1]` is y-direction flow

### Sparse Flow Model (XFeat)

The inference script generates three output files:

1. `*_matches.png` - Feature match visualization
   - Shows detected keypoints and matches between images
   - Lines connect matched features

2. `*_sparse_flow.png` - Sparse flow visualization
   - Shows flow vectors at detected keypoints
   - Arrows indicate motion direction and magnitude

3. `*_data.npz` - Raw data
   - Contains: keypoints0, keypoints1, matches, sparse flow vectors, confidence scores

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

### XFeat

- **GPU Mode** (recommended):
  - Input resolution 640×480: ~10-20ms (depending on GPU model)
  - **Extremely fast**: Up to 5x faster than traditional methods
  - Requires ~1-2GB VRAM
  - Sparse output: Computes flow only at keypoints (typically 1000-4000 points)

- **CPU Mode**:
  - Input resolution 640×480: ~100-300ms
  - Requires ~2-3GB RAM

**Note:** XFeat provides sparse (not dense) optical flow, making it ideal for:
- Real-time applications
- Feature tracking
- Visual odometry
- SLAM systems

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

**XFeat:**
```bibtex
@inproceedings{potje2024xfeat,
  title={XFeat: Accelerated Features for Lightweight Image Matching},
  author={Potje, Guilherme and Cadar, Felipe and Martins, Renato and Ferreira, Rares and Nascimento, Erickson R},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20542-20551},
  year={2024}
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

### XFeat
- [XFeat Official GitHub](https://github.com/verlab/accelerated_features)
- [XFeat Paper](https://arxiv.org/abs/2404.19174)
- [CVPR 2024 Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.pdf)

## License

This project is for learning and research purposes only. Please refer to the official repositories for the original model licenses.

## FAQ

### Q: Which model should I use?

A:
- **XFeat**: Fastest option (~10-20ms), sparse flow, ideal for feature tracking and real-time applications
- **GMFlow**: Fast dense flow (~50-100ms), good balance of speed and accuracy, single-pass inference
- **RAFT**: Most accurate dense flow (~80-150ms), iterative refinement allows trading speed for accuracy
- **Recommendation**:
  - Need speed and sparse flow is acceptable? → **XFeat**
  - Need dense flow with good speed? → **GMFlow**
  - Need highest accuracy dense flow? → **RAFT**

### Q: Can I run this without a GPU?

A: Yes! Use the `--device cpu` parameter to run on CPU, though it will be slower.

### Q: What image formats are supported?

A: Common image formats including PNG, JPG, JPEG, BMP, etc.

### Q: How to improve execution speed?

A:
1. Use GPU (CUDA)
2. For fastest results: Use XFeat (sparse flow)
3. For GMFlow: Already optimized for speed
4. For RAFT: Use `--small` flag or reduce `--iters` (e.g., 6 instead of 12)
5. Reduce input image resolution

### Q: What if I get a CUDA out of memory error?

A:
1. Reduce input image resolution
2. For RAFT: Use `--small` flag for RAFT-small model
3. Use CPU mode
4. Close other programs using GPU

## Contact

For questions or suggestions, please open an Issue for discussion.
