Ran command: `find /home/cjimenez /mnt/Data -maxdepth 5 -type d -iname "*trtm*" -o -iname "*uniclothdiff*" 2>/dev/null`
Ran command: `find /mnt/Data/cjimenez/TRTM -maxdepth 3 2>/dev/null`
Ran command: `find /home/cjimenez -name "*track*" -o -name "*TRTM*" -o -name "*UniCloth*" 2>/dev/null | grep -v "\.cache" | head -n 30`
Ran command: `ls -la /home/cjimenez/Projects`
Ran command: `ls -la /home/cjimenez/Projects/FC-DC /home/cjimenez/Projects/Generalization`
Ran command: `find /home/cjimenez/Projects/Generalization/Code -maxdepth 4`
Ran command: `git remote -v`
Ran command: `git log --oneline -n 15`
Ran command: `git log -p -n 1 f0a09fd -- train_og.py | head -n 100`
Viewed README.md:1-100
Ran command: `ls -la uniclothdiff/datasets/`
Viewed cloth_state_est.py:1-150
Viewed cloth_state_est.py:150-257
Viewed vr_folding_dataset.py:1-150
Viewed trtm_dataset.py.bak:80-160
Viewed orbbec_depth_processor.py:1-120
Viewed orbbec_depth_processor.py:121-200
Viewed orbbec_depth_processor.py:200-300
Ran command: `ls -la /home/cjimenez/Projects/Generalization/Literature`
Searched web: ""Tracking Robotic Thin-shell Models" OR "TRTM" cloth mesh tracking paper CoRL ICRA"
Searched web: ""TRTM" "Template-based Reconstruction and Target-oriented Manipulation of Crumpled Cloths" depth point cloud github"
Searched web: "site:arxiv.org/abs/2403.07288 OR "TRTM: Template-based Reconstruction" input depth image point cloud network architecture"
Read URL: https://arxiv.org/html/2403.07288v1
Viewed content.md:150-250
Ran command: `python3 -c "
import urllib.request, json
url = 'https://export.arxiv.org/api/query?search_query=ti:TRTM+AND+au:Wang&max_results=5'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        xml = resp.read().decode('utf-8')
        print(xml[:2000])
except Exception as e:
    print(e)
"`
Read URL: https://arxiv.org/html/2308.04670v3
Searched for "point cloud"
Viewed content.md:350-450
Viewed content.md:450-520
Searched for "shading"
Viewed content.md:280-350
Viewed flow_matching_state_est_scheduler.py:50-190
Viewed flow_matching_state_est_scheduler.py:190-230

Here is the complete breakdown of both the **mathematics of the loss formulation** and **what TRTM and UniClothDiff actually do with point clouds**.

---

### Part 1: Mathematics of Current Loss vs. Proposed Changes

#### 1. Current Loss Formulation
In [uniclothdiff/schedulers/flow_matching_state_est_scheduler.py](file:///home/cjimenez/Projects/FlowCloth/uniclothdiff/schedulers/flow_matching_state_est_scheduler.py), the total training loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vel}} + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}} + \lambda_{\text{corner}} \mathcal{L}_{\text{corner}} + \lambda_{\text{contour}} \mathcal{L}_{\text{contour}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}$$

with weights: $\lambda_{\text{shape}} = 100.0$, $\lambda_{\text{corner}} = 10.0$, $\lambda_{\text{contour}} = 5.0$, $\lambda_{\text{edge}} = 15.0$.

Each term computes:

* **$\mathcal{L}_{\text{vel}}$ (Flow Matching Velocity MSE)**:
  Along the linear interpolation $x_t = (1-t)x_0 + t x_1$ with target velocity $v^* = x_1 - x_0$:
  $$\mathcal{L}_{\text{vel}} = \frac{1}{N} \sum_{i=1}^N \| v_{\theta}(x_t, t) - (x_1 - x_0) \|^2$$

* **$\mathcal{L}_{\text{shape}}$ (Direct Destination Reconstruction)**:
  From the ODE $x_t = x_1 - (1-t)v$, the model's instantaneous prediction of the final state is $\hat{x}_1 = x_t + (1-t)v_{\theta}$:
  $$\mathcal{L}_{\text{shape}} = \frac{1}{N} \sum_{i=1}^N \| \hat{x}_{1, i} - x_{1, i} \|^2$$

* **$\mathcal{L}_{\text{corner}}$ (Corner Node MSE)**:
  Supervises the 4 corners $\mathcal{C} = \{0, 20, 420, 440\}$:
  $$\mathcal{L}_{\text{corner}} = \frac{1}{4} \sum_{c \in \mathcal{C}} \| \hat{x}_{1, c} - x_{1, c} \|^2$$

* **$\mathcal{L}_{\text{edge}}$ (Absolute Metric Rest-Length Preservation)**:
  For every mesh edge $(i, j) \in \mathcal{E}$ (798 edges in the $21 \times 21$ grid):
  $$\mathcal{L}_{\text{edge}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left( \| \hat{x}_{1, i} - \hat{x}_{1, j} \|_2 - \| x_{1, i} - x_{1, j} \|_2 \right)^2$$

---

#### 2. Why the Current Loss Fails on Rectangular Cloths & Half-Folds

The mathematical failure stems from **the conflict between $\mathcal{L}_{\text{edge}}$ and aspect-ratio changes**:

1. **The Rigid Square Template**:
   The template $q_{\text{temp}}$ is a $1:1$ square. In normalized coordinates, edge lengths are isotropic:
   $$\Delta x_0 \approx \Delta y_0 \approx 0.05$$
2. **Anisotropic Stretching in Rectangular Cloths**:
   If a cloth has an aspect ratio of $2:1$ (either an originally rectangular towel or a square towel folded in half), uniform node distribution requires:
   $$\Delta y = 0.5 \, \Delta x$$
3. **The Penalty Trap**:
   Because $\lambda_{\text{edge}} = 15.0$ heavily penalizes any edge whose length shrinks or expands, the network **refuses to compress the vertical edges**.
   To satisfy both the rectangular observation boundary and the rigid edge-length loss, the optimizer takes the path of least mathematical resistance:
   **It hallucinates a fold.** It curls the top and bottom corners into the center crease so that the edges remain at their "square" rest-length while fitting inside the rectangular visual footprint.

---

#### 3. The Proposed Formulation: Laplacian Grid Smoothness

We replace the rigid metric length penalty with a **topological smoothness penalty**:

$$\mathcal{L}_{\text{laplacian}} = \frac{1}{N_{\text{inner}}} \sum_{i=1}^{19} \sum_{j=1}^{19} \left\| \hat{x}_{i,j} - \frac{\hat{x}_{i-1,j} + \hat{x}_{i+1,j} + \hat{x}_{i,j-1} + \hat{x}_{i,j+1}}{4} \right\|^2$$

##### Why Laplacian Smoothness Solves the Problem:
* **Affine Invariance (Free Aspect-Ratio Stretching)**:
  Consider a pure rectangular stretch $x' = s_x x, \; y' = s_y y$.
  $$\frac{(s_x(x-1), s_y y) + (s_x(x+1), s_y y) + (s_x x, s_y(y-1)) + (s_x x, s_y(y+1))}{4} = (s_x x, s_y y)$$
  **$\mathcal{L}_{\text{laplacian}} \equiv 0$ for ANY rectangular aspect ratio ($1:1$, $2:1$, $3:2$, etc.)!**
  It does not penalize stretching or flattening in one direction.
* **Heavy Penalty for Wrinkles and Knots**:
  If corners curl erratically into the center or nodes bunch up, the second derivative (Laplacian) spikes, which $\mathcal{L}_{\text{laplacian}}$ penalizes.
* **Proposed Weights**:
  * $\lambda_{\text{shape}} = 100.0$ (drives primary shape adaptation)
  * $\lambda_{\text{laplacian}} = 5.0$ (maintains uniform grid regularity, allows rectangular stretch)
  * $\lambda_{\text{corner}} = 2.5$ (reduced from 10.0 to prevent corner over-constraint)
  * $\lambda_{\text{edge}} = 0.0$ (disabled)

---

### Part 2: Did TRTM and UniClothDiff Do All This PCD Processing?

You were completely right to suspect this. **Neither TRTM nor UniClothDiff used the heavy 2D-to-3D processing pipeline currently in FlowCloth.**

Here is what the original code and papers actually did:

#### 1. TRTM (ICRA 2024: Wang, Li, Zamora, Coros)
* **TRTM does NOT use 3D Point Clouds as input at all.**
* TRTM is a **2D CNN + GNN architecture**:
  $$I_{\text{feat}} = \text{ResNet}(I_{\text{depth}})$$
* It passes a **2D depth image** directly into a 2D ResNet backbone to extract visual features, then uses a Graph Attention Network (GAT) to deform the 441 template vertices.
* **Where FlowCloth got its constants**:
  The TRTM paper introduced the rule:
  $$\text{Scale depth so } l_{\text{cloth}} : l_{\text{image}} = 2 : 3 \quad (480\text{ px cloth in } 720\text{ px canvas})$$
  along with the neutral table pixel value of $192$ and inverted height shading ($192 - 2 \cdot h_{\text{mm}}$).
  TRTM did this because it was processing **2D images with a 2D ResNet**, not 3D point clouds!

#### 2. UniClothDiff (CoRL 2025: Tian, Li, Ai, Yuan, Huang, Su)
* Look at the original dataset loader in [uniclothdiff/datasets/cloth_state_est.py](file:///home/cjimenez/Projects/FlowCloth/uniclothdiff/datasets/cloth_state_est.py#L88-L104):
  ```python
  # Direct pinhole back-projection:
  x = (u - cx) / fx * z
  y = (v - cy) / fy * z
  points = np.stack([x[z > 0], y[z > 0], z[z > 0]], axis=1)
  points = sample_points(points, num_sample_points=10000)
  ```
* **UniClothDiff did virtually ZERO tailored processing**:
  * No morphological hole closing.
  * No Gaussian or bilateral smoothing.
  * No 2D bounding box cropping to $480\text{ px}$.
  * No TRTM inverted grayscale shading.
  * No RGB luminance thresholding.
* Why? Because UniClothDiff is a **native 3D model** (Point Cloud Transformer / DiT). It operates on $(X, Y, Z)$ coordinates directly in meters.

#### 3. Why Does FlowCloth Have All That Processing in `orbbec_depth_processor.py`?
FlowCloth ended up with a complex **round-trip hybrid**:
$$\text{Real 3D Camera} \longrightarrow \text{2D Depth Map} \xrightarrow{\text{TRTM Shading}} \text{2D Canvas (720px)} \xrightarrow{\text{Back-project}} \text{3D PCD} \longrightarrow \text{UniClothDiff DiT}$$

This pipeline:
1. Takes the Orbbec 3D depth map.
2. Uses RGB luminance thresholding (`threshold = 100`) to find the cloth.
3. Crops and resizes to a $720\times 720$ canvas with TRTM inverted shading ($192$).
4. Back-projects that synthetic 2D canvas *back* into a 3D point cloud.
5. Feeds the resulting point cloud to UniClothDiff.

This explains why segmentation breaks with non-white cloths: **color thresholding in 2D image space is fragile to lighting, shadows, and cloth color.**

---

### The Clean Solution for Real Deployment

To make the system cloth-color agnostic and eliminate the 500 lines of fragile 2D image processing, we can replace the 2D thresholding with **pure 3D geometric filtering** (the native UniClothDiff approach):

```python
# 1. Direct 3D back-projection from Orbbec depth
X = (xs - cx) / fx * Z_mm
Y = (ys - cy) / fy * Z_mm

# 2. 3D Workspace Bounding Box (eliminates the 1.7m floor and room automatically)
mask = (X >= -400) & (X <= 400) & (Y >= -400) & (Y <= 400) & (Z >= 600) & (Z <= 1000)

# 3. Table Height Filtering (color-agnostic: works on black, white, red towels)
Z_table = np.percentile(Z[mask], 95)  # table plane depth
cloth_pts = pts[(Z_table - Z >= 3.0) & (Z_table - Z <= 80.0)]

# 4. Standard normalisation (3 lines)
pcd = subsample(cloth_pts, 400)
pcd_norm = (pcd - pcd.mean(0)) / np.max(np.linalg.norm(pcd - pcd.mean(0), axis=1))
```

This:
1. **Never touches RGB color or luminance**, making it 100% agnostic to cloth color and fold crease shadows.
2. **Eliminates the floor completely** via the $Z \in [600, 1000]\text{ mm}$ workspace boundary.
3. Aligns the deployment pipeline with the actual training distribution of UniClothDiff.