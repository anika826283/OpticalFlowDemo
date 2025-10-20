# OpticalFlowDemo

基於 GMFlow 的光流估計示範專案

## 簡介

本專案實作了 GMFlow (Global Matching Flow) 的 inference 功能，用於估計連續影像之間的光流（Optical Flow）。GMFlow 是一個基於全局匹配的深度學習光流估計方法，發表於 CVPR 2022。

## 專案結構

```
OpticalFlowDemo/
├── gmflow/                    # GMFlow 模型實作
│   ├── __init__.py
│   ├── gmflow.py             # 主模型
│   ├── backbone.py           # CNN 特徵提取網路
│   ├── transformer.py        # Transformer 模組
│   └── geometry.py           # 幾何工具函數
├── data/
│   ├── input/                # 輸入影像目錄
│   └── output/               # 輸出結果目錄
├── checkpoints/              # 模型權重目錄
├── inference.py              # 主要的 inference 腳本
├── example.py                # 範例程式碼
├── utils.py                  # 工具函數
├── requirements.txt          # Python 依賴套件
└── README.md                 # 本文件
```

## 環境建置

### 系統需求

- Python 3.8 或以上
- CUDA 11.0 或以上（若使用 GPU）
- 4GB+ RAM（CPU 模式）或 4GB+ VRAM（GPU 模式）

### 步驟 1: 安裝 Python 環境

建議使用虛擬環境來管理依賴套件：

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 步驟 2: 安裝依賴套件

```bash
pip install -r requirements.txt
```

依賴套件包含：
- `torch>=2.0.0` - PyTorch 深度學習框架
- `torchvision>=0.15.0` - PyTorch 視覺工具
- `numpy>=1.21.0` - 數值計算
- `opencv-python>=4.5.0` - 影像處理
- `pillow>=9.0.0` - 影像 I/O
- `matplotlib>=3.5.0` - 視覺化
- `einops>=0.6.0` - 張量操作
- `timm>=0.9.0` - 視覺模型工具

### 步驟 3: （可選）下載預訓練權重

為了獲得最佳效果，建議下載官方預訓練權重：

1. 訪問 [GMFlow GitHub](https://github.com/haofeixu/gmflow)
2. 下載預訓練模型（例如：`gmflow_sintel-0c07dcb3.pth`）
3. 將權重檔案放置於 `checkpoints/` 目錄

```bash
# 範例（需要根據實際下載連結調整）
cd checkpoints
wget https://github.com/haofeixu/gmflow/releases/download/v0.1/gmflow_sintel-0c07dcb3.pth
cd ..
```

## 使用方法

### 方法 1: 使用命令列腳本

```bash
python inference.py \
    --img0 data/input/frame1.png \
    --img1 data/input/frame2.png \
    --checkpoint checkpoints/gmflow_sintel-0c07dcb3.pth \
    --output_dir data/output \
    --device cuda
```

參數說明：
- `--img0`: 第一張影像的路徑（必需）
- `--img1`: 第二張影像的路徑（必需）
- `--checkpoint`: 模型權重檔案路徑（可選，若不提供則使用隨機初始化權重）
- `--output_dir`: 輸出目錄（預設：`data/output`）
- `--device`: 運算裝置，`cuda` 或 `cpu`（預設：`cuda`）

### 方法 2: 使用範例程式

執行範例程式來測試模型（使用合成影像）：

```bash
python example.py
```

### 方法 3: 在程式碼中使用

```python
import torch
from gmflow import GMFlow
from utils import load_image, prepare_image_tensor, visualize_flow, pad_image, unpad_flow

# 初始化模型
model = GMFlow(
    num_scales=1,
    feature_channels=128,
    upsample_factor=8,
    num_transformer_layers=6,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# 載入影像
img0 = load_image('path/to/img0.jpg')
img1 = load_image('path/to/img1.jpg')

# 轉換為張量
img0_tensor = prepare_image_tensor(img0).to(device)
img1_tensor = prepare_image_tensor(img1).to(device)

# Padding
img0_padded, pad = pad_image(img0_tensor, divisor=8)
img1_padded, _ = pad_image(img1_tensor, divisor=8)

# 執行推理
with torch.no_grad():
    flow = model.inference(img0_padded, img1_padded)

# 移除 padding
flow = unpad_flow(flow, pad)

# 視覺化
flow_vis = visualize_flow(flow[0])
```

## 輸出說明

inference 腳本會產生兩個輸出檔案：

1. `*_flow_vis.png` - 光流的視覺化結果
   - 使用色彩編碼表示光流的方向和大小
   - 色調（Hue）表示方向
   - 亮度（Value）表示流動強度

2. `*_flow.npy` - 原始光流數據
   - NumPy 陣列格式
   - 形狀：`[2, H, W]`，其中 `[0]` 為 x 方向流動，`[1]` 為 y 方向流動

## 效能說明

- **GPU 模式**（推薦）：
  - 輸入解析度 640×480：約 50-100ms（視 GPU 型號而定）
  - 需要約 2-4GB VRAM

- **CPU 模式**：
  - 輸入解析度 640×480：約 2-5 秒
  - 需要約 4GB RAM

## 參考文獻

如果您使用本專案，請引用原始 GMFlow 論文：

```bibtex
@inproceedings{xu2022gmflow,
  title={GMFlow: Learning Optical Flow via Global Matching},
  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8121-8130},
  year={2022}
}
```

## 相關資源

- [GMFlow 官方 GitHub](https://github.com/haofeixu/gmflow)
- [GMFlow 論文](https://arxiv.org/abs/2111.13680)
- [CVPR 2022 論文](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)

## 授權

本專案僅供學習和研究使用。原始 GMFlow 模型的授權請參考官方 repository。

## 常見問題

### Q: 沒有 GPU 可以執行嗎？

A: 可以！使用 `--device cpu` 參數即可在 CPU 上執行，但速度會較慢。

### Q: 支援哪些影像格式？

A: 支援常見的影像格式，包括 PNG, JPG, JPEG, BMP 等。

### Q: 如何提升執行速度？

A:
1. 使用 GPU（CUDA）
2. 降低輸入影像解析度
3. 使用預訓練權重（可能影響準確度）

### Q: 遇到 CUDA out of memory 錯誤怎麼辦？

A:
1. 降低輸入影像解析度
2. 使用 CPU 模式
3. 關閉其他使用 GPU 的程式

## 聯絡資訊

如有問題或建議，歡迎開啟 Issue 討論。
