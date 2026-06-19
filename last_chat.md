Good. Now I have the full picture. Here's the honest answer:

Yes, it will work — but there's a sim-to-real gap you need to bridge.

At inference, the model needs:

q_temp — the flat template. This is fixed and known a priori (just load the pickle once). No sensor needed.
pcd — a [400, 3] point cloud of the cloth in the current frame.
The PCD is cross-attended, not concatenated with the mesh. PointcloudEmbed runs FPS + KNN grouping on the raw PCD → a patch embedding → cross-attention in every transformer block. This means the model is conditioned on the shape of the point cloud, not its cardinality or ordering, which is important for occlusion robustness.

Occluded / folded states: Yes, this is the core value proposition. Because the model denoises the full mesh state (all 441 / 80 nodes) conditioned on a partial observation (PCD sees only the visible surface), it can hallucinate occluded cloth geometry via the learned prior. The template provides topological structure so the model knows node connectivity even when nodes are invisible.

The real gap you need to bridge:

The model was trained on PCDs sampled from mesh vertices (clean, full-coverage, no perspective occlusion). A real depth camera gives you:

Only the top-visible surface (strong occlusion for folded cloth)
Real sensor noise (structured light / ToF artifacts)
Missing regions (specular reflections, cloth borders)
A different coordinate frame
You'll need a depth-to-PCD preprocessing step for the real robot:


# pseudocode for real deploymentdepth = camera.get_depth()          # [H, W]pcd_raw = backproject(depth, K)     # [N, 3] using camera intrinsicspcd_cloth = segment(pcd_raw, mask)  # isolate cloth pointspcd = normalize_and_downsample(pcd_cloth, n=400)  # COM-center, unit scale, 400 ptspred_mesh = pipeline(pcd, q_temp)   # → [441, 3] full mesh
The normalization step (COM-center + unit scale) must exactly match training convention — that's the most critical part to get right. The TRTM real/test images are per-frame normalized uint8, which is why the _getitem_real path exists but is only qualitative — for actual metric-depth cameras (RealSense, ZED, etc.) you get proper metric depth and can do real backprojection.

Whether the model transfers well depends on how much the sim PCDs resemble the real ones after normalization — training on TRTM (which was designed for sim-to-real) should help significantly versus training only on VR-Folding.

Claude Sonnet 4.6 • 1x