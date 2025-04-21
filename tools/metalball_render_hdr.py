""" Blender script to render hdri lights.

This script is used to render images of hdri lights with a metal ball. It takes 
in the path to .hdr files and renders images of this light by reflecting from the 
metal ball. The images are saved to the output directory.

Example usage:
    blender -b -P metalball_render_hdr.py -- \
        --output_dir ./output \
        --light light.hdr \

Please notify the parameters light_theta / segments / ring_count for render quality
"""

from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import argparse
import math
import os
import sys
import bpy
from mathutils import Vector,Euler

def add_light(light_type="AREA", energy=1000, point=[0.,0.5,0.]):
    # add a new light
    bpy.ops.object.light_add(type=light_type.upper(),location=point)
    light2 = bpy.data.objects[light_type.capitalize()]
    light2.data.energy = energy
    # light2.rotation_euler = Euler((math.radians(-90), 0, 0), 'XYZ')

def add_hdri(path, angle = Euler([0.,0.,0.]), bg = Vector([1.,1.,1.,1.])):
    world_node_tree = bpy.context.scene.world.node_tree
    world_node_tree.nodes.clear()
    environment_texture_node = world_node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    environment_texture_node.image = bpy.data.images.load(path)
    hdri_background_node = world_node_tree.nodes.new(type='ShaderNodeBackground')
    world_output_node = world_node_tree.nodes.new(type='ShaderNodeOutputWorld')
    mapping_node = world_node_tree.nodes.new(type='ShaderNodeMapping')
    mapping_node.inputs['Rotation'].default_value = angle
    texture_coordinate_node = world_node_tree.nodes.new(type='ShaderNodeTexCoord')
    # white_background_node = world_node_tree.nodes.new(type='ShaderNodeBackground')
    # white_background_node.inputs['Color'].default_value = bg
    # mix_shader_node = world_node_tree.nodes.new(type='ShaderNodeMixRGB')
    # light_path_node = world_node_tree.nodes.new(type='ShaderNodeLightPath')

    world_node_tree.links.new(texture_coordinate_node.outputs['Generated'],mapping_node.inputs['Vector'])
    world_node_tree.links.new(mapping_node.outputs['Vector'],environment_texture_node.inputs['Vector'])
    world_node_tree.links.new(environment_texture_node.outputs['Color'],hdri_background_node.inputs['Color'])
    
    world_node_tree.links.new(hdri_background_node.outputs['Background'],world_output_node.inputs['Surface'])

    # world_node_tree.links.new(hdri_background_node.outputs['Background'],mix_shader_node.inputs['Color1'])
    # world_node_tree.links.new(white_background_node.outputs['Background'],mix_shader_node.inputs['Color2'])
    # world_node_tree.links.new(light_path_node.outputs['Is Camera Ray'],mix_shader_node.inputs['Fac'])
    # world_node_tree.links.new(mix_shader_node.outputs['Color'],world_output_node.inputs['Surface'])

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def render_hdr_with_metalball(args):
    reset_scene()

    ### modify segments and ring_count for the ball quality
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0), segments=args.quality, ring_count=args.quality//2)
    metal_ball = bpy.context.active_object
    metal_ball.name = "MetalBall"
    create_metal_material(metal_ball, roughness=0.0)

    normalize_scene()
    
    if args.light.endswith('.hdr'):
        ### modify light theta for the position in hdr
        light_theta = - math.pi / 2
        add_hdri(path=args.light, angle=Euler([0.,0.,light_theta]))
    else:
        if args.light == 'top':
            point = [0.5, 0.0, 2.0]
        elif args.light == 'right':
            point = [1.0, 2.0, 0.0]
        elif args.light == 'right_top':
            point = [1.0, 2.0, 1.0]
        elif args.light == 'right_bottom':
            point = [1.0, 2.0, -1.0]
        else:
            point = [2.0, 0.0, 0.0]
        add_light(light_type='point', energy = 500, point = point)

    cam, cam_constraint = setup_camera()
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    phi = theta = math.radians(0)
    point = (
        1.5 * math.cos(phi) * math.cos(theta),
        1.5 * math.cos(phi) * math.sin(theta),
        1.5 * math.sin(phi),
    )
    cam.location = point
    os.makedirs(args.output_dir, exist_ok=True)
    light_name = os.path.splitext(os.path.basename(args.light))[0]
    render_path = os.path.join(args.output_dir,f'{light_name}.png')
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)

def create_metal_material(obj, roughness=0.0, metallic=1.0):
    mat = bpy.data.materials.new(name="MetalMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic

    output = nodes.new(type='ShaderNodeOutputMaterial')

    links = mat.node_tree.links
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

if __name__ == "__main__":
    
    context = bpy.context
    scene = context.scene

    render = scene.render
    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100
    render.film_transparent = True

    scene.use_nodes = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = 1024
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = 'Standard'

    parser = argparse.ArgumentParser()
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--quality', type=int, default=2048)
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    render_hdr_with_metalball(args)