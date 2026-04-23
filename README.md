# Python MNIST Classification

這是一個使用 PyTorch 實現的簡單 MNIST 手寫數字辨識專案。

## 功能特點

- 使用卷積神經網路 (CNN) 進行影像分類。
- 自動下載並處理 MNIST 資料集。
- 支援 CPU 與 GPU (CUDA) 加速訓練。
- 使用 `uv` 進行現代化的 Python 套件管理。

## 專案結構

- `main.py`: 包含模型定義、訓練與測試邏輯的核心程式碼。
- `pyproject.toml`: 專案依賴與配置資訊。
- `data/`: 存放 MNIST 資料集的目錄（已在 git 中排除）。

## 快速開始

### 1. 安裝環境

建議使用 [uv](https://github.com/astral-sh/uv) 來安裝依賴：

```bash
uv sync
```

### 2. 執行訓練

使用以下指令開始訓練模型：

```bash
uv run main.py
```

## 模型架構

模型採用簡單的 CNN 架構：
- 2 層卷積層 (Convolutional Layers)
- 2 層最大池化層 (Max Pooling)
- 2 層全連接層 (Fully Connected Layers)
- 使用 ReLU 激活函數與 Log Softmax 輸出

## 依賴項

- Python >= 3.14
- PyTorch
- Torchvision
- Matplotlib
