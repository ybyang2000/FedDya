# FedDya Data Generation Support

This script provides data generation and highly controllable Non-IID partitioning support for **FedDya**. 

It utilizes a dual Dirichlet Distribution to precisely control both Quantity Skew and Label Skew across clients, tailored for complex Federated Multi-Task Learning (MTL) experiments.

## ✨ Core Features

- **Dual Non-IID Control**:
  - `alpha1`: Controls the data quantity skew across clients (imbalance in dataset size).
  - `alpha2`: Controls the label skew within each individual client (imbalance in categories).

## 📊 Supported Scenarios (4 Datasets)

1. **Multi-Attribute MNIST (Colored & Rotated)**
   - Contains labels across three dimensions: digit classification, foreground color recognition, and rotation angle recognition.
2. **Paired CIFAR-10**
   - Horizontally concatenates an "animal" image and a "vehicle" image. Designed for dual-branch/dual-task networks.
3. **Four-Corner Mosaic CIFAR-100**
   - Extracts sub-classes from 4 specified coarse superclasses (large mammals, small/medium mammals, household electrical devices, household furniture).
   - Assembles them into a 64x64 four-corner mosaic image. 
4. **FLAME Dataset (UAV Aerial Imagery)**
   - Supports aerial imagery for multi-task scenarios: scene recognition and wildfire monitoring.
   - **Note**: Requires the pre-processed FLAME dataset files.

## 💾 Dataset Access (Large Files)

Due to GitHub's file size limitations, the processed **FLAME** dataset is hosted on Google Drive to ensure full reproducibility.

- **Download Link**: [Download flame.zip](https://drive.google.com/file/d/11qQX7QQC9SkY22D_bOplELnZxk-m3N8Z/view?usp=sharing)
- **Instructions**: Please download and extract the `flame.zip` archive into your local `data/` directory before running the generation scripts.

## 🛠️ Requirements

```bash
pip install torch torchvision numpy Pillow
