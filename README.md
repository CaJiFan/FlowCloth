# FlowCloth: Real-Time Cloth State Estimation 👕

FlowCloth is a real-time cloth state estimation and tracking system built on top of the powerful [UniClothDiff](https://github.com/Jiayuan-Gu/torkit3d) framework. It extends UniClothDiff's diffusion-based generative state estimation models to work live with **Orbbec RGB-D cameras**.

## ✨ Key Contributions

This project introduces several new capabilities for real-world, live deployment:
- **Real-Time Orbbec Camera Integration**: Captures live RGB-D streams using `pyorbbecsdk`.
- **Live Inference GUI (`orbbec_frame.py`)**: A master calibration and inference GUI that handles live depth hole-filling, float-smoothing, and cloth segmentation.
- **Dynamic TRTM Shading**: Implements real-time Tensor-based Real-Time Modeling (TRTM) inverted shading and background masking.
- **3D-to-2D Live Projection**: Subsamples the live point cloud, runs the UniClothDiff inference pipeline, and projects the predicted mesh nodes/edges back onto the live color feed.
- **Automated Intrinsics & Depth Processing**: Custom modules (`orbbec_depth_processor.py`, `get_orbbec_intrinsics.py`) to handle physical camera calibration and coordinate conversions.

## 🚀 Getting Started with FlowCloth

### 1. Download Checkpoints
First, download the pre-trained FlowCloth checkpoints from our public Google Drive:
👉 [Download Checkpoints Here](https://drive.google.com/drive/folders/1m2wHp1xFJAZTXVzA88lES3nXIrJ0yVrJ?usp=sharing)

### 2. Installation (with ROS 2 Support)
To use the live Orbbec camera and send commands to the Kinova arm via ROS 2, you must configure your Python environment to inherit your system's ROS 2 installation (e.g., ROS 2 Humble). 

```bash
# 1. Source your global ROS 2 installation
source /opt/ros/humble/setup.bash

# 2. Create the virtual environment WITH system packages enabled (critical for rclpy)
python3 -m venv .venv --system-site-packages

# 3. Activate the environment and install ML dependencies (uv pip recommended)
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Live Deployment, ROS Bridges & Benchmarking
All deployment commands live in **[deployment/README.md](deployment/README.md)**. It covers:
- the Orbbec calibration / overlay GUIs
- live Kinova inference (FlowCloth and the TRTM baseline)
- ROS 1 / ROS 2 publishing via the UDP bridges
- rollout recording and the offline rollout benchmark


---

## 📖 Original UniClothDiff Reference

*FlowCloth is based on UniClothDiff. Below is the original documentation for training and data collection.*

### UniClothDiff: Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation

**Abstract:** Cloth manipulation is challenging due to its highly complex dynamics, near-infinite degrees of freedom, and frequent self-occlusions...

#### Installation
```bash
conda env create -f environment.yml
conda activate clothdiff

pip install "git+https://github.com/Jiayuan-Gu/torkit3d.git"

# Install the customized diffusers fork used by the repo
cd third_party/diffusers
pip install .

# Install UniClothDiff
pip install -e .
```

#### Citation
```bibtex
@article{tian2025uniclothdiff,
  author    = {Tian, Tongxuan and Li, Haoyang and Ai, Bo and Yuan, Xiaodi and Huang, Zhiao and Su, Hao},
  title     = {Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

**License:** MIT License
