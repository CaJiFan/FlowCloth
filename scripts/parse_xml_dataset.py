import xml.etree.ElementTree as ET
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glob import glob

def parse_cloth_xml_sequence(xml_path):
    """Parses the XML into a [Frames, 400, 3] PyTorch tensor."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    vertices = []
    for pos in root.findall('.//position'):
        if pos.get('x') is None or pos.get('y') is None or pos.get('z') is None:
            continue
        vertices.append([float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))])
        
    flat_mesh_tensor = torch.tensor(np.array(vertices, dtype=np.float32))
    
    total_points = flat_mesh_tensor.shape[0]
    if total_points % 400 != 0:
        raise ValueError(f"Total points ({total_points}) is not a multiple of 400!")
        
    num_frames = total_points // 400
    return flat_mesh_tensor.view(num_frames, 400, 3)

def animate_sequence(seq_tensor, output_filename="cloth_animation_centered.gif"):
    """Creates a 3D wireframe animation of the cloth sequence from a Top-Down view."""
    print(f"Generating Centered Top-Down animation for {seq_tensor.shape[0]} frames...")
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Extract all coordinates
    all_x = seq_tensor[:, :, 0].numpy()
    all_y = seq_tensor[:, :, 1].numpy()
    all_z = seq_tensor[:, :, 2].numpy()
    
    # 2. Find the exact mathematical center of the entire sequence
    mid_x = (all_x.max() + all_x.min()) / 2.0
    mid_y = (all_y.max() + all_y.min()) / 2.0
    mid_z = (all_z.max() + all_z.min()) / 2.0
    
    # 3. Find the maximum distance it travels in ANY direction to create a perfect cube
    max_range = max(all_x.max() - all_x.min(), 
                    all_y.max() - all_y.min(), 
                    all_z.max() - all_z.min()) / 2.0
    
    margin = 0.05
    xlim = [mid_x - max_range - margin, mid_x + max_range + margin]
    ylim = [mid_y - max_range - margin, mid_y + max_range + margin]
    zlim = [mid_z - max_range - margin, mid_z + max_range + margin]

    def update_graph(frame_idx):
        ax.clear() 
        
        # Lock the axes to our PERFECT CUBE boundaries
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Centered Top-Down Sequence (Frame {frame_idx}/{seq_tensor.shape[0]})")
        
        # Lock camera to top-down view
        ax.view_init(elev=90, azim=-90)
        
        # Grab current frame and reshape
        frame_data = seq_tensor[frame_idx].numpy()
        X = frame_data[:, 0].reshape(20, 20)
        Y = frame_data[:, 1].reshape(20, 20)
        Z = frame_data[:, 2].reshape(20, 20)
        
        # Plot wireframe
        ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.8, alpha=0.7)
        
        return fig,

    # Create the animation
    ani = animation.FuncAnimation(fig, update_graph, frames=seq_tensor.shape[0], interval=50, blit=False)
    
    ani.save(output_filename, writer='pillow', fps=20)
    print(f"Done! Animation saved to: {output_filename}")

if __name__ == "__main__":
    data_dir = "../../datasets/VR_Folding/Dataset_02/"
    
    for xml_file in glob(f'{data_dir}/*.xml'):
        try:
            sequence = parse_cloth_xml_sequence(xml_file)
            animate_sequence(sequence, output_filename=f"{data_dir}/animations/{xml_file.split('/')[-1]}.gif")
        except Exception as e:
            print(f"Error in {xml_file}: {e}")