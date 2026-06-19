import numpy as np

# 1. Load your original 400 points
vertices = []
with open("./assets/vr_template_d1_400.obj", "r") as f:
    for line in f:
        if line.startswith("v "):
            parts = line.strip().split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
vertices = np.array(vertices)

# 2. Find the 76 perimeter indices
grid_size = 20
contour = []
for i in range(grid_size):
    for j in range(grid_size):
        if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1:
            contour.append(i * grid_size + j)

# 3. Extract and save the new 76-point mesh
edge_vertices = vertices[contour]

with open("vr_template_76.obj", "w") as f:
    for v in edge_vertices:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")

print(f"Saved vr_template_76.obj with {len(edge_vertices)} vertices!")