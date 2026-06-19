import numpy as np
import pickle

def create_edge_template(orig_path, out_path):
    print(f"Loading original template from: {orig_path}")
    
    with open(orig_path, 'rb') as f:
        template = pickle.load(f)

    print(f"Original template['mesh_pos'].shape: {template['mesh_pos'].shape}")

    # 1. Recreate the 80-point boundary extraction logic
    grid_size = 21
    indices = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    top = indices[0, :]
    bottom = indices[-1, :]
    left = indices[1:-1, 0]
    right = indices[1:-1, -1]
    
    # Ordered 1D loop of the 80 boundary indices
    contour_idx = np.concatenate([top, right, bottom[::-1], left[::-1]])

    # 2. Initialize the new template
    new_template = {}

    # Extract the 80 boundary vertices
    new_template['mesh_pos'] = template['mesh_pos'][contour_idx]
    
    # Safely handle target_pos (fine vs coarse)
    if 'target_pos' in template:
        if len(template['target_pos']) == 441:
            new_template['target_pos'] = template['target_pos'][contour_idx]
        else:
            print(f"Note: target_pos has size {len(template['target_pos'])} (coarse). Adjusting to new patch size.")

    # 3. Build Fine Edges (80-node 1D Ring)
    edges_src = []
    edges_dst = []
    for i in range(80):
        edges_src.extend([i, (i + 1) % 80])
        edges_dst.extend([(i + 1) % 80, i])
    new_template['edge_idx'] = np.array([edges_src, edges_dst])

    # 4. Remove faces
    new_template['face_idx'] = np.empty((3, 0), dtype=np.int64)

    # ==========================================
    # 5. PATCHIFY LOGIC (16 patches, 5 points each)
    # ==========================================
    num_patches = 16
    points_per_patch = 5

    # group_vtx_idx: The 3D coordinates of the 16 patch centers
    new_template['group_vtx_idx'] = new_template['mesh_pos'][2::points_per_patch]

    # If target_pos was coarse, assign it to our new coarse nodes to prevent shape mismatches
    if 'target_pos' in template and len(template['target_pos']) != 441:
        new_template['target_pos'] = new_template['group_vtx_idx']

    # group_dense_vtx_idx: Maps 80 fine points to 16 patches
    new_template['group_dense_vtx_idx'] = np.repeat(np.arange(num_patches), points_per_patch)

    # group_edge_idx: 1D Ring connecting the 16 patches
    coarse_src = []
    coarse_dst = []
    for i in range(num_patches):
        coarse_src.extend([i, (i + 1) % num_patches])
        coarse_dst.extend([(i + 1) % num_patches, i])
    new_template['group_edge_idx'] = np.array([coarse_src, coarse_dst])
    
    # group_dense_edge_idx: Bipartite edges between 80 fine and 16 coarse nodes
    dense_edges_src = np.arange(80) 
    dense_edges_dst = new_template['group_dense_vtx_idx'] 
    
    new_template['group_dense_edge_idx'] = np.array([
        np.concatenate([dense_edges_src, dense_edges_dst]), 
        np.concatenate([dense_edges_dst, dense_edges_src])
    ])

    # 6. Save the new perfectly formatted template
    with open(out_path, 'wb') as f:
        pickle.dump(new_template, f)
        
    print(f"Successfully saved 80-point edge template to: {out_path}")
    print(f" - mesh_pos shape: {new_template['mesh_pos'].shape}")
    print(f" - group_vtx_idx (patches) shape: {new_template['group_vtx_idx'].shape}")

if __name__ == "__main__":
    original_template_path = "../TRTM/datasets/template_square/template_square.pickle" 
    new_template_path = "../TRTM/datasets/template_square/template_square_edge.pickle"
    
    create_edge_template(original_template_path, new_template_path)