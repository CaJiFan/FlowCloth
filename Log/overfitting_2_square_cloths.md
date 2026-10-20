Ran command: `python3 -c "
import glob, numpy as np

files = sorted(glob.glob('/mnt/Data/cjimenez/TRTM/datasets/template_square/simu/train/*.simu_mesh.txt'))
corners = [0, 20, 440, 420] # BL, BR, TR, TL

# Let's search for clean half-folds where BL, BR fold near TL, TR (or y goes from -1 to 0 or +1 to 0)
half_folds = []
for i in range(min(5000, len(files))):
    mesh = np.loadtxt(files[i])
    c = mesh[corners]
    # Check if either bottom edge is near middle or top edge is near middle
    bl_y, br_y, tr_y, tl_y = c[:, 1]
    # In canonical square, y in [-1, 1]
    # Bottom folded in half: bl_y > -0.2 and br_y > -0.2 and tl_y > 0.8 and tr_y > 0.8
    if bl_y > -0.3 and br_y > -0.3 and tl_y > 0.7 and tr_y > 0.7:
        half_folds.append((i, 'bottom_folded_up', [round(x, 2) for x in c[:, 1]]))
    elif tl_y < 0.3 and tr_y < 0.3 and bl_y < -0.7 and br_y < -0.7:
        half_folds.append((i, 'top_folded_down', [round(x, 2) for x in c[:, 1]]))

print(f'Found {len(half_folds)} clean half-folds out of 5000 samples.')
for h in half_folds[:10]:
    print(h)
"`

The background task finished. Let's inspect the output of task-1353!

Used tool: manage_task
Used tool: manage_task

These three screenshots tell the entire scientific story with crystal clarity:

1. **SS1 (Originally rectangular cloth)**: The old checkpoint (`sw100.3`) adapts to the rectangle well because it has no rigid edge penalty forbidding $\Delta x \ne \Delta y$.
2. **SS3 (Single-corner fold)**: Works beautifully because the diagonal fold produces an unmistakably asymmetric pentagon.
3. **SS2 (Horizontal half-fold)**: **The core failure mode**: The cloth is folded in half, but the model spreads all 441 nodes flat over the rectangle (`TL`/`TR` at top, `BL`/`BR` at bottom), completely ignoring the fact that one layer is folded on top of the other!

---

### Why Did `sw100.3` Fail in SS2, While the New Loss Curled Inward?

You are caught between two opposite extremes:

| Loss Setup | On Rectangular Cloth (SS1) | On Half-Fold (SS2) | Why? |
| :--- | :--- | :--- | :--- |
| **Old Checkpoint** (`shape=100`, `corner=0`, `edge=0`) | **Good**: Stretches smoothly into a rectangle. | **Fails (Lazily flat)**: Spreads the template flat, completely misses the fold. | Pure coordinate MSE takes the path of least resistance: spreading points flat across the visible rectangle satisfies MSE with zero fold awareness. |
| **New Checkpoint** (`shape=100`, `corner=10`, `edge=15`) | **Fails**: Curled `TL` inward because it refused to stretch. | **Fails (Collapsed)**: Both corners tried to meet in the middle crease. | $\mathcal{L}_{\text{edge}} = \text{MSE}(\|\hat{e}\|, L_0)$ is an **absolute metric length penalty**. It strictly forces edges to remain $0.05$ (square), penalizing rectangular deformation. |

---

### Why Did the Model in SS2 Miss the Fold Crease? (The Normalization Gap)

Look at SS2 on the left (depth canvas) vs. right (prediction):
* In the depth map, **the horizontal fold crease is clearly visible** as a dark horizontal seam.
* Why did the Transformer ignore it?
  Look at how the point cloud is normalized in `orbbec_depth_processor.py`:
  $$\text{scale} = \max(\|\text{pts}_{\text{centered}}\|)$$
  For a $40 \times 20\text{ cm}$ cloth, $\text{scale} \approx 250\text{ mm}$.
  The $(X, Y)$ coordinates range from $-1.0$ to $+1.0$ (span of $2.0$).
  The physical fold step in $Z$ is $\approx 2\text{ mm}$ of cloth thickness.
  In normalized coordinates:
  $$Z_{\text{crease}} = \frac{2\text{ mm}}{250\text{ mm}} = \mathbf{0.008}$$
  **$0.008$ is 250 times smaller than $(X, Y)$!** To the Transformer’s cross-attention, the point cloud appears $99.6\%$ completely flat. Since $(X, Y)$ looks like a $2:1$ rectangle, the model treats it as a flat rectangular cloth!

---

### The "Forgiving" Formulation: How to Let the Flow Matching Field Adapt

To allow the velocity field to stretch for rectangles **while still capturing folds and sharp corners**, we need a loss formulation that is **aspect-ratio agnostic**:

#### 1. Replace Absolute Edge Loss with Laplacian Grid Smoothness ($\mathcal{L}_{\text{laplacian}}$)
Instead of forcing every edge to equal the square rest length $0.05$ (which broke rectangular adaptation):
$$\mathcal{L}_{\text{laplacian}} = \frac{1}{N} \sum_{i,j} \left\| x_{i,j} - \frac{x_{i-1,j} + x_{i+1,j} + x_{i,j-1} + x_{i,j+1}}{4} \right\|^2$$
* **Why this works**: On a uniform $2:1$ rectangle, the second spatial derivative along the grid is **zero**! A rectangular stretched mesh has **zero penalty**.
* It only penalizes unnatural crinkles, jagged kinks, or bunched-up knots, while freely allowing the cloth to be a square, a $2:1$ rectangle, or a $3:2$ rectangle!

#### 2. Ratio-Preserving Asymmetric Strain (Soft Upper Bound)
If you still want an inextensibility prior so folds don't unrealistically over-stretch, use an **asymmetric upper-bound hinge**:
$$\mathcal{L}_{\text{strain}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \text{ReLU}\left( \frac{\|\hat{x}_i - \hat{x}_j\|}{L_{ij}^{(0)}} - \gamma \right)^2$$
* Only penalizes edges if they stretch more than $\gamma$ (e.g. $\gamma = 1.5$, allowing up to $50\%$ aspect-ratio stretch).
* Has **zero penalty on compression** (cloth folds by compressing in the projected plane).

#### 3. Moderate Corner Weight ($\lambda_{\text{corner}} = 2.0\text{--}3.0$, not $10.0$)
* $\lambda_{\text{corner}} = 10.0$ was so dominant that the model would distort the whole mesh just to force corners into local minima.
* A weight of $2.0\text{--}3.0$ acts as a gentle anchor that resolves the 4-corner ambiguity without tearing the mesh.

#### 4. Boost the Normalized Z Scale (Make the Fold Visible)
In `orbbec_depth_processor.py` (and in dataset `_depth_to_pcd`):
* Give $Z$ (height relative to table) an amplification factor (e.g., $3\times\text{--}5\times$) or normalize $Z$ by its own standard deviation rather than the $40\text{ cm}$ horizontal table extent.
* Once the $2\text{ mm}$ crease step produces a clear gradient in the input point cloud tokens, the cross-attention layers can easily differentiate a 2-layer fold from a flat single-layer rectangle!

---

### Summary Recommendation for Your Next Training Run

In your training config:
```yaml
diffusion_cfg:
  type: FlowMatching_StateEst
  shape_loss_weight: 100.0        # Global reconstruction
  corner_loss_weight: 2.5          # Gentle corner guidance (not 10.0)
  contour_loss_weight: 1.0         # Boundary alignment
  edge_loss_weight: 0.0            # Disable rigid metric edge loss!
  laplacian_loss_weight: 10.0      # Scale-free smoothness (allows rectangular stretching)
```
This directly resolves the conflict: it lets the mesh stretch freely into rectangles (SS1) while giving the corners and folds the guidance they need (SS2, SS3).