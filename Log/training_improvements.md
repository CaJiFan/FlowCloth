Searched web: ""Template-based Reconstruction and Target-oriented Manipulation of Crumpled Cloths" "GNN" "depth" "occlusion" OR "visibility" OR "template""
Searched web: ""TRTM: Template-based Reconstruction and Target-oriented Manipulation of Crumpled Cloths" arxiv "input" OR "point cloud" OR "depth""
Searched web: ""2308.04670" "template" "depth" "GNN" "reconstruction""
Searched web: ""2308.04670" "visibility" "loss" OR "edge" OR "reconstruction""
Read URL: https://ar5iv.labs.arxiv.org/html/2308.04670
Searched for "loss"
Viewed content.md:25-35

Here is a detailed analysis addressing each of your questions with a view toward your **ICRA submission**.

---

### 1. How many loss terms will I end up having? Should I get rid of the shape loss or just add them?

#### What your current `loss_shape` actually is
In [`flow_matching_state_est_scheduler.py`](file:///home/cjimenez/Projects/FlowCloth/uniclothdiff/schedulers/flow_matching_state_est_scheduler.py#L92-L107):
$$\mathcal{L}_{\text{vel}} = \|v_{\text{pred}} - v_{\text{target}}\|^2$$
$$\mathcal{L}_{\text{shape}} = \|\hat{x}_1 - x_1\|^2 = (1-t)^2 \|v_{\text{pred}} - v_{\text{target}}\|^2$$

$\mathcal{L}_{\text{shape}}$ is not an actual geometric or physical shape constraint. It is simply an unweighted coordinate MSE scaled by $(1-t)^2$. 

#### Recommendation: Do NOT throw it away, but formulate a clean multi-objective loss
In an ICRA paper, you do not want an unwieldy mess of 10 ad-hoc losses. A well-principled, defensible loss formulation consists of **3 distinct components**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FM}} + \lambda_{\text{shape}} \mathcal{L}_{\text{recon}} + \lambda_{\text{boundary}} \mathcal{L}_{\text{boundary}} + \lambda_{\text{physics}} \mathcal{L}_{\text{physics}}$$

1. **Flow Matching Velocity Loss ($\mathcal{L}_{\text{FM}}$)**:
   The core generative objective supervising the vector field $v_t$.
2. **Global Reconstruction Anchor ($\mathcal{L}_{\text{recon}}$)**:
   Your current `loss_shape` (weighted MSE over all 441 vertices). This ensures global placement and broad surface fit.
3. **Boundary / Corner Loss ($\mathcal{L}_{\text{boundary}}$)**:
   $$\mathcal{L}_{\text{boundary}} = \frac{\alpha_{\text{corner}}}{4} \sum_{c \in \text{Corners}} \|\hat{x}_c - x_{gt, c}\|^2 + \frac{\alpha_{\text{contour}}}{|\mathcal{C}|} \sum_{i \in \mathcal{C}} \|\hat{x}_i - x_{gt, i}\|^2$$
   * Solves the **"4 vs 441" gradient dilution problem**: The 4 corners represent only $0.9\%$ of vertices. Without this boost, their gradient is drowned out by the 437 interior points.
4. **Physical Inextensibility Loss ($\mathcal{L}_{\text{physics}}$)**:
   Cloth is virtually inextensible. For all mesh edges $(i, j) \in \mathcal{E}$:
   $$\mathcal{L}_{\text{physics}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left( \|\hat{x}_i - \hat{x}_j\| - L_{ij}^{(0)} \right)^2$$
   * **Why this is critical for hidden/inner folds**: When a corner is folded underneath, it has zero observed depth points. An inextensibility loss physically penalizes the hidden fabric from collapsing into a shrunken blob under the fold.

---

### 2. How does this fit with the Contour-FlowCloth approach? Scientific advice for an ICRA paper

#### Does a contour/corner loss work for both?
**Yes, absolutely.**
* **For Full Mode (441 points)**: $\mathcal{L}_{\text{boundary}}$ acts as an **anchor prior** that forces the outer perimeter and corners to stay accurate while the interior fills in.
* **For Contour Mode (80 points)**: $\mathcal{L}_{\text{boundary}}$ is naturally the primary objective (with a corner weight $\alpha_{\text{corner}}$ on the 4 corner indices within the 80 points).

#### Strategic Advice for an ICRA Submission
An ICRA robotics paper requires a strong **robotics manipulation justification**, not just computer vision metrics. The story you can present is compelling:

1. **The Downstream Manipulation Reality**:
   * Robot grippers (e.g., Kinova, YuMi) almost **never grasp flat interior cloth points**—they target **corners and edges** (e.g., dynamic flinging, corner-pulling, edge-grasping).
   * Therefore, an estimation error of $5\text{ cm}$ at a corner is catastrophic for manipulation, whereas an error of $5\text{ cm}$ on an interior flat point is harmless.
2. **The Paper Narrative**:
   * **Problem**: Standard diffusion/flow-matching cloth estimators optimize global point-cloud MSE, which inherently sacrifices high-curvature boundaries and corners due to vertex-count dominance ($99\%$ interior vs $1\%$ corners).
   * **Contribution 1**: Boundary-aware and physically-constrained Flow Matching ($\mathcal{L}_{\text{boundary}} + \mathcal{L}_{\text{physics}}$) that maintains fidelity on occluded corners.
   * **Contribution 2**: **Contour-FlowCloth** as an ultra-fast, boundary-focused policy (80 points vs 441) achieving real-time inference speeds suitable for closed-loop visual servoing and dynamic manipulation.
3. **The Ablation Table**:
   * Baseline FM (unweighted MSE)
   * + Boundary/Corner loss
   * + Inextensibility physics loss
   * Contour-only mode vs. Full mode (Trade-off: latency vs. full-surface topology).

---

### 3. The Sim2Real Gap: What did TRTM actually do in their paper?

#### Did TRTM sample from the "privileged" 3D mesh vertices?
**No, they did not.**
The `_mesh_to_pcd` sampling in your codebase was inherited from **VR-Folding** (which sampled 400 random vertices from `simu_mesh.txt` for fast dataloading).

Here is how the original TRTM authors (Wang et al., ICRA 2024) actually handled depth images, occlusion, and the sim-to-real gap:

1. **They fed the rendered 2D Depth Image directly into a ResNet**:
   * TRTM did not feed a synthetic 3D point cloud. They took the rendered $552 \times 552$ (or normalized $720 \times 720$) top-down depth map and extracted image features via a ResNet-18 backbone.
   * Because the input was a single-view depth image, **their network was exposed to real self-occlusion from day 1**.
2. **Explicit Keypoint Loss**:
   * In Section III-C of the TRTM paper, they explicitly supervised **9 landmarks**: the **4 corners**, the **4 edge midpoints**, and the **1 center vertex** ($L_{\text{key}, p}$). They found this was essential to prevent permutation and corner-folding errors.
3. **Unidirectional Chamfer Loss from the Observed Depth**:
   * TRTM introduced a unidirectional Chamfer loss $\mathcal{L}_{\text{cham}}$ from the **observed depth points $\hat{D}$ to the predicted mesh vertices $\tilde{P}$**:
     $$\mathcal{L}_{\text{cham}} = \frac{1}{|\hat{D}|} \sum_{d \in \hat{D}} \min_{p \in \tilde{P}} \|d - p\|$$
   * Notice the direction: **every observed depth point must be explained by a mesh vertex**, but mesh vertices on the underside of a fold are **not** penalized if they don't have depth points!
4. **Edge-Length Regularization ($\mathcal{L}_{\text{regu}}$)**:
   * TRTM added an explicit edge regularization loss preventing the synthetic mass-spring edges from stretching or collapsing under occluded layers.
5. **Visibility Prediction Head**:
   * Each mesh vertex predicted a binary flag $f_i \in \{0, 1\}$ indicating whether that vertex was currently visible from the top-view or hidden underneath an occluded layer.

---

### Next Steps

To close this gap in FlowCloth:
1. **In Training (`trtm_dataset.py`)**: Introduce self-occlusion into `_mesh_to_pcd` (e.g., sampling point clouds from `simu_depth.png` or raycasting/Hidden Point Removal), so the network learns to infer folded corners from partial top-surface boundaries instead of expecting privileged 3D points inside folds.
2. **In Scheduler / Loss**: Add the explicit 4-corner loss term and edge-length inextensibility penalty.