import cv2
import numpy as np
import trimesh
import xatlas
from PIL import Image, ImageFilter
import argparse

# Barycentric interpolation
def barycentric_interpolate(v0, v1, v2, c0, c1, c2, p):
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        return (c0 + c1 + c2) / 3
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    interpolate_color = u * c0 + v * c1 + w * c2
    return np.clip(interpolate_color, 0, 255)

# Point-in-Triangle test
def is_point_in_triangle(p, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1])

    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def convert(input_mesh_path, output_mesh_path, scale, resolution):
    mesh = trimesh.load(input_mesh_path)
    vertex_colors = mesh.visual.vertex_colors
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    vertices = mesh.vertices[vmapping]
    vertex_colors = vertex_colors[vmapping]

    mesh.vertices = vertices
    mesh.faces = indices
    texture_size = resolution

    buffer_size = texture_size * scale

    texture_buffer = np.zeros((buffer_size, buffer_size, 4), dtype=np.uint8)

    # Texture-filling loop
    for face in mesh.faces:
        uv0, uv1, uv2 = uvs[face]
        c0, c1, c2 = vertex_colors[face]

        uv0 = (uv0 *(buffer_size - 1)).astype(int)
        uv1 = (uv1 *(buffer_size - 1)).astype(int)
        uv2 = (uv2 *(buffer_size - 1)).astype(int)

        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), buffer_size - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), buffer_size - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                if is_point_in_triangle(p, uv0, uv1, uv2):
                    color = barycentric_interpolate(uv0, uv1, uv2, c0, c1, c2, p)
                    texture_buffer[y, x] = np.clip(color, 0, 255).astype(
                        np.uint8
                    )
    # Inpainting
    image_bgra = texture_buffer.copy()
    mask = (image_bgra[:, :, 3] == 0).astype(np.uint8)* 255
    image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
    inpainted_bgr = cv2.inpaint(
        image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    inpainted_bgra = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2BGRA)
    texture_buffer = inpainted_bgra[::-1]
    image_texture = Image.fromarray(texture_buffer)

    # Median filter
    image_texture = image_texture.filter(ImageFilter.MedianFilter(size=3))

    # Gaussian blur
    image_texture = image_texture.filter(ImageFilter.GaussianBlur(radius=1))

    # Downsample
    image_texture = image_texture.resize((texture_size, texture_size), Image.LANCZOS)

    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        baseColorTexture=image_texture,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )

    # material = trimesh.visual.material.SimpleMaterial(image=image_texture)
    visuals = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh.visual = visuals
    mesh.export(output_mesh_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()
    convert(input_mesh_path=args.input, output_mesh_path=args.output, scale=args.scale, resolution=args.resolution)