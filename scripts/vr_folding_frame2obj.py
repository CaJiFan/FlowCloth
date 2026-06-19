import xml.etree.ElementTree as ET

def export_flat_obj(xml_path, output_filename="./assets/vr_template_d1_400.obj"):
    print(f"Parsing {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 1. Extract the 400 Canonical Vertices (Frame 0)
    vertices = []
    for pos in root.findall('.//position'):
        if pos.get('x') is None: continue
        vertices.append([float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))])
    
    # 2. Extract the 722 Faces
    faces = []
    for face in root.findall('.//face'):
        # .obj files are 1-indexed, so we add 1 to every vertex index!
        v1 = int(face.get('v1')) + 1
        v2 = int(face.get('v2')) + 1
        v3 = int(face.get('v3')) + 1
        faces.append([v1, v2, v3])
        
    # 3. Write standard .obj file format
    with open(output_filename, 'w') as f:
        # Write 400 vertices
        for v in vertices[:400]:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # Write 722 faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            
    print(f"Success! Saved {output_filename}. You can now pass this to tmpl_patchify.py!")

if __name__ == "__main__":
    data_dir = "../../datasets/VR_Folding/Dataset_01/"
    xml_file = f'{data_dir}/01_1LM_2PCM_01.xml'
    export_flat_obj(xml_file)