""" Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in the path to 
.glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P render_blender.py -- \
        --object_path example.glb \
        --output_dir ./output \
        --light_path light.hdr \
        --engine CYCLES \
        --mode object \
        --num_images 40 \

"""

from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import argparse
import math
import os
import random
import sys
import time
import urllib.request
import numpy as np
import bpy
from mathutils import Vector,Euler

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

def build_rgb_background(world: bpy.types.World,
                         rgb: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
                         strength: float = 1.0) -> None:
    world.use_nodes = True
    node_tree = world.node_tree

    rgb_node = node_tree.nodes.new(type="ShaderNodeRGB")
    rgb_node.outputs["Color"].default_value = rgb

    node_tree.nodes["Background"].inputs["Strength"].default_value = strength

    node_tree.links.new(rgb_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

def reset_light():
    for l in ['Light','Area','Point','Sun','Spot']:
        if l in bpy.data.objects.keys():
            bpy.data.objects[l].select_set(True)
            bpy.ops.object.delete()

def add_light(light_type="AREA", energy=1000, point=[0.,0.5,0.]):
    # add a new light
    bpy.ops.object.light_add(type=light_type.upper(),location=point)
    light2 = bpy.data.objects[light_type.capitalize()]
    light2.data.energy = energy
    # light2.rotation_euler = Euler((math.radians(-90), 0, 0), 'XYZ')

def add_hdri(path, strength = 1.0, angle = Euler([0.,0.,0.]), bg = Vector([1.,1.,1.,1.])):
    world_node_tree = bpy.context.scene.world.node_tree
    world_node_tree.nodes.clear()
    environment_texture_node = world_node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    environment_texture_node.image = bpy.data.images.load(path)
    hdri_background_node = world_node_tree.nodes.new(type='ShaderNodeBackground')
    hdri_background_node.inputs['Strength'].default_value = strength
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

def load_vertex_color_obj(obj_path):
    vertices = []
    vertex_colors = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 6:
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    vertices.append((x, y, z))
                    vertex_colors.append((r, g, b))
            
            elif line.startswith('f '):
                faces.append([int(i.split('/')[0]) - 1 for i in line.split()[1:]])

    vertices, vertex_colors, faces = np.array(vertices), np.array(vertex_colors), faces
    
    mesh = bpy.data.meshes.new("CustomMesh")
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new("VertexColorObj", mesh)
    bpy.context.collection.objects.link(obj)
    
    color_layer = mesh.vertex_colors.new(name="Col")
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            color_layer.data[loop_idx].color = (*vertex_colors[vert_idx], 1.0)
    
    mat = bpy.data.materials.new(name="VertexColorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    vertex_color_node = nodes.new('ShaderNodeVertexColor')
    vertex_color_node.layer_name = "Col"
    
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    
    mat.node_tree.links.new(vertex_color_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    mat.node_tree.links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])
    obj.data.materials.append(mat)
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)

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

def normalize_obj(obj):
    minX,minY,minZ,maxX,maxY,maxZ = 0,0,0,0,0,0
    for v in obj.data.vertices:
        vx,vy,vz = obj.matrix_world @ v.co
        if vx < minX:
            minX = v.co[0]
        if vx > maxX:
            maxX = v.co[0]
        if vy < minY:
            minY = v.co[1]
        if vy > maxY:
            maxY = v.co[1]
        if vz < minZ:
            minZ = v.co[2]
        if vz > maxZ:
            maxZ = v.co[2]
    center = Vector([maxX+minX,maxY+minY,maxZ+minZ])/2
    scale = 2/max([maxX-minX,maxY-minY,maxZ-minZ])
    print(scale)
    for v in obj.data.vertices:
        v.co = (v.co-center)*scale

def save_images(args) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    if args.vc:
        load_vertex_color_obj(args.object_path)
    else:
        load_object(args.object_path)
    
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    obj = mesh_objects[-1]
    # normalize_obj(obj)
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (math.radians(args.rotation_x),
                          math.radians(args.rotation_y),
                          math.radians(args.rotation_z))
    
    normalize_scene()

    cam, cam_constraint = setup_camera()
    ### create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    for i in range(args.num_images):
        if args.mode == 'light':
            light_theta = i / args.num_images * math.pi * 2
        elif args.mode == 'object':
            light_theta = - math.pi / 2

        phi = theta = math.radians(0)
        cam.location = (
            args.camera_dist * math.cos(phi) * math.cos(theta),
            args.camera_dist * math.cos(phi) * math.sin(theta),
            args.camera_dist * math.sin(phi),
        )

        if args.light_path:
            s = 0.8 if 'outdoor' in args.light_path else 1.0
            add_hdri(path = args.light_path, strength = s, angle = Euler([0.,0.,light_theta]))
        else:
            reset_light()
            scale = 2 if 'Trellis' in args.object_path or 'hunyuan' in args.object_path else 1
            energy = 400 * scale

            if args.light_type == 'top':
                add_light(light_type='point', energy = energy, point = [0.5,0,2])

            elif args.light_type == 'right':
                add_light(light_type='point', energy = energy, point = [1,2,0])

            elif args.light_type == 'right_top':
                cam.rotation_euler = (math.radians(30), 0, 0)
                cam.location[2] = 0.5
                add_light(light_type='point', energy = energy, point = [1,2,1])
            
            elif args.light_type == 'right_bottom':
                cam.rotation_euler = (math.radians(-30), 0, 0)
                cam.location[2] = -0.5
                add_light(light_type='point', energy = energy, point = [1,2,-1])
            
        render_path = os.path.join(args.output_dir, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        if args.mode == 'object':
            obj.rotation_euler.z -= math.pi * 2 / args.num_images

def add_background(args):
    scene = bpy.context.scene
    node_tree = scene.node_tree
    node_tree.nodes.clear()
    image_node_1 = node_tree.nodes.new(type='CompositorNodeImage')
    image_node_2 = node_tree.nodes.new(type='CompositorNodeImage')
    alpha_over_node = node_tree.nodes.new(type='CompositorNodeAlphaOver')
    composite_node = node_tree.nodes.new(type='CompositorNodeComposite')
    node_tree.links.new(image_node_1.outputs['Image'], alpha_over_node.inputs[1])
    node_tree.links.new(image_node_2.outputs['Image'], alpha_over_node.inputs[2])
    node_tree.links.new(alpha_over_node.outputs['Image'], composite_node.inputs['Image'])
    for i in os.listdir(args.output_dir):
        image = bpy.data.images.load(os.path.join(args.output_dir,i))
        bg = bpy.data.images.new('bg', args.H, args.W)
        bg.pixels = np.ones(args.H * args.W * 4, dtype=np.float32)
        image_node_1.image = bg
        image_node_2.image = image
        scene.render.filepath = os.path.join(args.output_dir,i)
        bpy.ops.render.render(write_still=True)
    # os.system(f'convert -loop 0 -delay 10 {args.output_dir}/*.png _.gif')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=str, required=True, help="Path to the object file")
    parser.add_argument('--light_path', type=str, help='Path to the light file')
    parser.add_argument('--light_type', type=str, choices=["top", "right", "right_top", "right_bottom"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--mode", type=str, default="object", choices=["light", "object"])
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--rotation_x", type=int, default=0)
    parser.add_argument("--rotation_y", type=int, default=0)
    parser.add_argument("--rotation_z", type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--bg', action='store_true')
    parser.add_argument('--vc', action='store_true')
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--camera_dist", type=int, default=1.5)
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    assert args.light_path or args.light_type

    context = bpy.context
    scene = context.scene

    render = scene.render
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = args.H
    render.resolution_y = args.W
    render.resolution_percentage = 100
    render.film_transparent = not args.bg

    scene.use_nodes = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = 'Standard'
    
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        if 'CPU' in d["name"]:
            d["use"] = args.cpu
        else:
            d["use"] = not args.cpu
        print(d["name"], d["use"])

    object_name = os.path.splitext(os.path.basename(args.object_path))[0]

    args.output_dir = os.path.join(args.output_dir, object_name)
    try:
        start_i = time.time()
        save_images(args)
        end_i = time.time()
        print("Finished", args.object_path, "in", end_i - start_i, "seconds")
    except Exception as e:
        with open('e.log','a') as f:
            f.write('*'*10)
            f.write('\n')
            f.write(args.object_path)
            f.write('\n')
            f.write(str(e))
            f.write('\n')
        print("Failed to render",args.object_path)
        print(e)