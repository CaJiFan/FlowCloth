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

### Requirements
Make sure you have installed the Orbbec SDK (`pyorbbecsdk`). Then, follow the original UniClothDiff installation below.

### Running Live Inference
Connect your Orbbec camera and run the master GUI:

```bash
# Run the live GUI with an existing checkpoint
python scripts/orbbec_frame.py \
  --ckpt_dir path/to/your/checkpoint \
  --mode edge \
  --num_steps 10
```

Inside the GUI:
1. Adjust the **Zoom** and **Threshold** sliders until the cloth fits properly in the green target box.
2. The UI will display the live RGB, the TRTM shaded depth, and the live diffusion model predictions overlaid on the cloth.
3. Press `S` to save a calibration frame, or `Q` / `ESC` to quit.

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
