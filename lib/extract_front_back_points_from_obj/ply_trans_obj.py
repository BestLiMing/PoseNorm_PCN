import trimesh
import os
import numpy as np
import imageio.v2 as imageio


def ply_to_obj_with_trimesh(ply_path, obj_path):
    mesh = trimesh.load(ply_path)  # read ply file
    mesh.export(obj_path)  # export obj


def export_obj_with_texture(mesh_path, obj_path, texture_path):
    mesh = trimesh.load(mesh_path)
    output_dir = os.path.dirname(obj_path)
    os.makedirs(output_dir, exist_ok=True)

    texture_name = os.path.basename(texture_path)
    output_texture = os.path.join(output_dir, texture_name)
    with open(texture_path, 'rb') as src, open(output_texture, 'wb') as dst:
        dst.write(src.read())
    mtl_content = f"""
    newmtl texture_material
    map_Kd {texture_name}
    """

    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    with open(mtl_path, 'w') as f:
        f.write(mtl_content.strip())

    with open(obj_path, 'w') as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in mesh.visual.uv:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        f.write("usemtl texture_material\n")
        for face in mesh.faces:
            indices = [f"{i + 1}/{i + 1}" for i in face]
            f.write(f"f {' '.join(indices)}\n")


def add_texture_to_obj(obj_path, texture_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mesh = trimesh.load(obj_path)
    try:
        texture_image = imageio.imread(texture_path)
    except ImportError:
        from PIL import Image
        texture_image = np.array(Image.open(texture_path))
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        vertices = mesh.vertices
        uv = vertices[:, [0, 1]]
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv = (uv - uv_min) / (uv_max - uv_min + 1e-6)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

    mesh.visual.material = trimesh.visual.material.SimpleMaterial(
        name="scan_tex",
        image=texture_image
    )

    output_obj = os.path.join(output_dir, os.path.basename(obj_path))
    mesh.export(output_obj, include_texture=True)

    output_texture = os.path.join(output_dir, os.path.basename(texture_path))
    if not os.path.exists(output_texture):
        with open(texture_path, 'rb') as src, open(output_texture, 'wb') as dst:
            dst.write(src.read())


def obj_to_ply(obj_file: str):
    mesh = trimesh.load_mesh(obj_file)
    ply_file = obj_file.replace('.obj', '.ply')
    mesh.export(ply_file)
    return ply_file


