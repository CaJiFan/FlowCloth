# FlowCloth Deployment

Scripts for running FlowCloth live with an Orbbec RGB-D camera and publishing cloth corners to ROS for the Kinova arm. The folder also has tools to record real-robot rollouts and benchmark models on them offline.

## Prerequisites

- **Environment:** follow the installation steps in the [root README](../README.md#2-installation-with-ros-2-support) (`.venv` created with `--system-site-packages`). The [Dockerfile](../Dockerfile) builds a ready-made ROS 2 Humble image.
- **Run from the repo root.** The scripts import each other as top-level modules, so call them as `python3 deployment/<script>.py ...`.
- **FlowCloth / UniClothDiff checkpoints:** put each experiment in `checkpoints/<exp>/` with `config.yml` and `checkpoints/checkpoint-*`. Download links are in the [root README](../README.md#1-download-checkpoints). The scripts read `config.yml` to tell FM from DDPM and full from edge mode.
- **TRTM baseline checkpoint:** `checkpoint_large_size.pt`. Pass it with `--ckpt` / `--trtm_ckpt`; the default path is `/home/cjimenez/Downloads/checkpoint_large_size.pt`. Its template and edges are in `assets/trtm_large/`.

## 1. Calibration and visual debugging

Use the overlay GUIs to check the camera setup and tracking before you involve the robot.

```bash
# FlowCloth / UniClothDiff, single model
python3 deployment/orbbec_overlay_3d.py --ckpt_dir checkpoints/<exp> --mode edge --num_steps 10

# Full and edge models side by side
python3 deployment/orbbec_overlay_3d.py \
  --ckpt_dir_full checkpoints/<full_exp> --ckpt_dir_edge checkpoints/<edge_exp> --num_steps 10

# TRTM GNN baseline
python3 deployment/trtm_orbbec_overlay.py --ckpt /path/to/checkpoint_large_size.pt
```

In the GUI:
1. Adjust the **Zoom** and **Threshold** sliders until the cloth fits in the green target box.
2. The windows show the live RGB, the TRTM-shaded depth and the predicted mesh overlaid on the cloth.
3. Press `S` to save a calibration frame (to `trtm_data_calibrated_<timestamp>/`). Press `Q` / `ESC` to quit.

Useful flags: `--ema_alpha` (temporal smoothing, default `0.3`), `--max_depth` (mm cutoff to drop the floor), `--disable_plane`, `--disable_pruning` and `--tta_rotations 4` (test-time yaw rotations). `trtm_orbbec_overlay.py` also accepts the ROS flags from [section 3](#3-ros-publishing-and-udp-bridges).

To print the camera's factory intrinsics:
```bash
python3 deployment/get_orbbec_intrinsics.py
```

## 2. Live inference (Kinova)

The inference scripts capture frames, estimate the cloth state, extract the 4 corners and optionally publish them to ROS.

```bash
# FlowCloth / UniClothDiff (FM vs DDPM and edge vs full are auto-detected from config.yml)
python3 deployment/kinova_cloth_inference.py --exp_dir checkpoints/<exp> --ros --ema_alpha 0.3

# TRTM GNN baseline
python3 deployment/trtm_cloth_inference.py --ckpt /path/to/checkpoint_large_size.pt --ros
```

- **Topic:** `/cloth_corners` (`geometry_msgs/PoseArray`). Each pose holds one corner in metres, in the camera frame (`--ros_frame`, default `camera_depth_optical_frame`). Your `tf2` tree handles the camera-to-base transform. To publish in the robot base frame instead, pass `--cam_to_base` with the 16 row-major values of the 4×4 transform.
- **`--ema_alpha`** sets temporal smoothing: `0.1` is smooth but laggy, `0.9` is fast but jittery, and `1.0` turns it off.
- **`--arm_depth_offset 60`–`100`** masks out the robot arm: cloth pixels that are this many mm closer to the camera than the cloth surface get inpainted. The default `0` disables it.
- **`--num_steps`** sets the number of sampling steps. It defaults to 10 for FM and 100 for DDPM. `--step <int>` loads a specific checkpoint instead of `latest`.
- **`--no_gui`** runs headless. `--save_dir <dir>` saves per-frame `.npz` files (pcd, prediction, corners).

Both scripts also run offline on saved frames:
```bash
python3 deployment/kinova_cloth_inference.py --exp_dir checkpoints/<exp> \
  --depth_png trtm_data_calibrated_<ts>/depth_raw_00000.png \
  --color_png trtm_data_calibrated_<ts>/color_00000.png      # add --loop to repeat
```

## 3. ROS publishing and UDP bridges

`--ros` first tries to publish natively from the venv with `rospy`. If that import fails, it falls back to sending JSON over UDP to `127.0.0.1:9876`. This happens, for example, when the venv's Python doesn't match the system ROS Python (3.10 vs 3.8). A bridge running under the system Python then republishes the data to ROS.

- `kinova_cloth_inference.py` falls back to UDP automatically.
- `trtm_cloth_inference.py` and `trtm_orbbec_overlay.py` also accept `--udp` to force the UDP path. They accept `--udp_port` too, but the bridges are hard-coded to `9876`.

Start the bridge in a **separate terminal with the venv not activated**:

```bash
# ROS 1 (Noetic): publishes /cloth_corners, plus /cloth_mesh (PointCloud2) when a mesh is sent
source /opt/ros/noetic/setup.bash
python3 deployment/ros1_udp_bridge.py

# ROS 2 (Foxy, or another ROS 2 distro): publishes /cloth_corners
source /opt/ros/foxy/setup.bash
python3 deployment/ros2_udp_bridge.py
```

Then start the inference in the venv terminal, e.g. `python3 deployment/kinova_cloth_inference.py --exp_dir checkpoints/<exp> --ros`. Check the output with `rostopic echo /cloth_corners` (ROS 1) or `ros2 topic echo /cloth_corners` (ROS 2).

## 4. Recording rollouts

Both inference scripts can record real-robot rollouts to `rollouts/rollout_YYYYMMDD_HHMMSS/`. A rollout contains `depth/`, `color/`, `rollout_data.npz`, live metrics (`metrics_summary.json`, `metrics_per_frame.csv`) and a video.

```bash
python3 deployment/kinova_cloth_inference.py --exp_dir checkpoints/<exp> --ros --record_rollout   # records from start to finish
```
Without `--record_rollout`, press `R` in the overlay window to start or stop a recording. `--rollout_dir` changes the output folder.

The live metrics come from the older `rollout_recorder.py`. For reported numbers, use the offline benchmark below.

## 5. Offline benchmark on recorded rollouts

### 5.1 Label corners (optional, needed for the corner-error metric)

```bash
python3 deployment/label_corners.py --rollout rollouts/<rollout> --every 20
```
Click every visible corner and skip the ones hidden by folds or the arm. Corner identity doesn't matter because matching is permutation-invariant. Labels are saved to `<rollout>/corner_labels.json`.
Controls: left click adds a corner, right click removes the nearest one, `SPACE`/`n` saves and moves to the next frame, `b` goes back, `c` clears the frame, and `q`/`ESC` saves and quits.

### 5.2 Run the benchmark

```bash
python3 deployment/benchmark_rollouts.py --models trtm ddpm@100 fm_full@4 fm_edge@4 \
  --rollouts_glob "rollouts/rollout_20260915_*" --output_dir evaluation_reports/benchmark_replay_<date>
```

- **Models** are given as `<key>@<steps>`. The keys are `trtm`, `ddpm`, `fm_full` and `fm_edge`. Their checkpoint directories are set in `MODEL_REGISTRY` at the top of `benchmark_rollouts.py`.
- **Inputs:** each frame is preprocessed **once** (seeded) with the live calibration (`--seg_threshold 120 --crop_size 480`), so every model sees identical inputs and ground truth.
- **Metrics** (from `rollout_metrics.py`):
  - **Ground truth:** the observed arm-free cloth surface and silhouette. Predicted surfaces are resampled to a fixed density.
  - **Accuracy:** p2o / o2p / Chamfer, F1@10/20 mm, silhouette IoU, boundary Chamfer and corner error (labelled frames only).
  - **Plausibility:** perimeter, area and edge-length CV.
  - **Temporal:** jitter.
  - **Latency:** batch-1 latency and FPS.
  - **Latency-aware replay:** uses the recorded frame timestamps.
- **Outputs:** `summary_metrics.{json,csv}`, `latex_table.tex`, `per_rollout.csv`, `per_frame/` and `predictions/` (cached `.npz`), plus best/worst videos in `videos/`.

Useful flags:
- `--reuse_predictions` recomputes metrics and videos from the cached predictions without running inference.
- `--videos 0` skips the videos.
- `--max_frames N` / `--dry_run` are for quick checks.
- `--fixed_noise` reuses the same initial noise for every frame.
- `--ema_alpha` applies live-style smoothing before the metrics are computed.

### 5.3 Inspect rollouts

```bash
python3 deployment/rollout_inspector_gui.py --rollout_dir rollouts/<rollout>   # defaults to the latest rollout
```
Scrub frames with the slider or `A`/`D`. Other keys:
- `Space`: play
- `M`: cycle models
- `V`: cycle views (RGB, depth, dual, side-by-side)
- `G`: ground-truth points
- `H`: hide the HUD (for figures)
- `S`: screenshot
- `Q`: quit

## 6. Other tools

```bash
# Browse TRTM dataset samples with model predictions overlaid (no camera needed)
python3 deployment/trtm_overlay_gui.py --ckpt_dir checkpoints/<exp>/checkpoints/checkpoint-best \
  --data_dir /path/to/TRTM/template_square --split val --mode full --num_steps 5

# Run the depth preprocessing on one saved frame
python3 deployment/orbbec_depth_processor.py --depth trtm_data_calibrated_<ts>/depth_raw_00000.png \
  --color trtm_data_calibrated_<ts>/color_00000.png
```

## File reference

| File | Purpose |
|---|---|
| `orbbec_overlay_3d.py` | Live calibration and overlay GUI for FlowCloth / UniClothDiff |
| `trtm_orbbec_overlay.py` | Same GUI for the TRTM baseline |
| `kinova_cloth_inference.py` | Live/offline FlowCloth inference, ROS corner publisher, rollout recording |
| `trtm_cloth_inference.py` | Same pipeline for the TRTM baseline |
| `trtm_gnn_model.py` | Pure-PyTorch TRTM GNN model (no `torch_scatter`) |
| `orbbec_depth_processor.py` | Depth → normalised point cloud (`normalise_pcd` must stay in sync with training normalization) |
| `get_orbbec_intrinsics.py` | Print the Orbbec camera intrinsics |
| `ros1_udp_bridge.py` / `ros2_udp_bridge.py` | UDP → ROS 1 / ROS 2 republishers (port 9876) |
| `rollout_recorder.py` | Live rollout recording and the older live metrics |
| `rollout_metrics.py` | Offline benchmark metrics (no I/O) |
| `benchmark_rollouts.py` | Offline replay benchmark → `evaluation_reports/` |
| `label_corners.py` | Hand-label corners for the corner-error metric |
| `rollout_inspector_gui.py` | Scrub through recorded rollouts and compare models |
| `trtm_overlay_gui.py` | Browse the TRTM dataset with prediction overlays |
