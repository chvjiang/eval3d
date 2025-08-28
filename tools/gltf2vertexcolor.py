""" Script to convert gltf objects to vertex colored objects.

This script is used to convert gltf objects whose texture can not be 
read by kiuitools. It takes in a .glb file and convert it to a .obj 
file with vertex colors.

Example usage:
    python gltf2vertexcolor.py
        --input example.glb
        --output example.obj
"""

import numpy as np
import trimesh
from pygltflib import GLTF2
import argparse


def create_trimesh(gltf):
    mesh = gltf.meshes[0]
    primitive = mesh.primitives[0]
    assert hasattr(primitive.attributes, 'COLOR_0')
    accessor_index = primitive.attributes.COLOR_0
    accessor = gltf.accessors[accessor_index]
    buf = gltf.bufferViews[accessor.bufferView]
    colors = np.frombuffer(
        gltf.binary_blob()[buf.byteOffset:buf.byteOffset + buf.byteLength],
        dtype=np.uint8,
    ).reshape(-1, 4)

    positions_accessor = primitive.attributes.POSITION
    positions = gltf.accessors[positions_accessor]
    buf = gltf.bufferViews[positions.bufferView]
    vertices = np.frombuffer(
        gltf.binary_blob()[buf.byteOffset:buf.byteOffset + buf.byteLength],
        dtype=np.float32
    ).reshape(-1, 3)

    indices_accessor = primitive.indices
    indices = gltf.accessors[indices_accessor]
    buf = gltf.bufferViews[indices.bufferView]
    faces = np.frombuffer(
        gltf.binary_blob()[buf.byteOffset:buf.byteOffset + buf.byteLength],
        dtype=np.int32
    ).reshape(-1, 3)

    trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh_obj.visual.vertex_colors = colors
    return trimesh_obj

def main(input_glb, output_glb):
    gltf = GLTF2().load(input_glb)
    trimesh_obj = create_trimesh(gltf)
    trimesh_obj.export(output_glb)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    main(args.input, args.output)