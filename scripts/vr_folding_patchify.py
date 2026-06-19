import xml.etree.ElementTree as ET
import numpy as np
import pickle
import torch

def create_vr_template_pickle(xml_path, output_filename="vr_template_400.pickle"):
    print(f"Parsing {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 1. Extract the 400 Canonical Vertices (Frame 0)
    vertices = []
    for pos in root.findall('.//position'):
        if pos.get('x') is None: continue
        vertices.append([float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))])
    
    # Grab just the first 400 points (Frame 0) and center them!
    mesh_pos = np.array(vertices[:400], dtype=np.float32)
    mesh_pos = mesh_pos - mesh_pos.mean(axis=0) # Mean-center it!
    
    # 2. Extract the 722 Faces (Triangles)
    faces = []
    for face in root.findall('.//face'):
        v1 = int(face.get('v1'))
        v2 = int(face.get('v2'))
        v3 = int(face.get('v3'))
        faces.append([v1, v2, v3])
    
    faces = np.array(faces, dtype=np.int32)
    
    # 3. Save to Pickle format expected by UniClothDiff Patchifier
    template_dict = {
        'mesh_pos': mesh_pos, # [400, 3]
        'faces': faces        # [722, 3]
    }
    
    with open(output_filename, 'wb') as f:
        pickle.dump(template_dict, f)
        
    print(f"Success! Saved {output_filename}")
    print(f"Mesh shape: {mesh_pos.shape}")
    print(f"Faces shape: {faces.shape}")

if __name__ == "__main__":
    # Point to any of the sequence XMLs you downloaded
    create_vr_template_pickle("01_1LM_2PCM_01.xml")