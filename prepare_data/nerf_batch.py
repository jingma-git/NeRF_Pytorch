# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os, json
import numpy as np
import bpy

obj_max_size = 2.5
lamp1_energy = 1.0
# lamp2_energy = 0.015
lamp2_energy = 1.0
lamp3_energy = 1.0
lamp4_energy = 1.0

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def load_obj(obj_file_path):
    global bbox3D
    bpy.ops.import_scene.obj(filepath=obj_file_path)

    # meshes joined
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH':
            ob.select = True
            bpy.context.scene.objects.active = ob
            obj_name = ob.name
        else:
            ob.select = False
    bpy.ops.object.join()
    ob = bpy.context.object

    # adjust obj size
    ori_x = bpy.data.objects[obj_name].dimensions[0]
    ori_y = bpy.data.objects[obj_name].dimensions[1]
    ori_z = bpy.data.objects[obj_name].dimensions[2]
    print("orig object dimensions:", bpy.data.objects[obj_name].dimensions)

    scale = obj_max_size / max([ori_x, ori_y, ori_z])
    bpy.data.objects[obj_name].dimensions = [ori_x * scale, ori_y * scale, ori_z * scale]
    print("object dimensions:", bpy.data.objects[obj_name].dimensions)
    bbox3D = bpy.data.objects[obj_name].dimensions

    # put the obj at center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.data.objects[obj_name].location[0] = 0
    bpy.data.objects[obj_name].location[1] = 0
    bpy.data.objects[obj_name].location[2] = 0
    return obj_name


# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}


parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=100,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--split', type=str, default='train',
                    help='Train, val, test dataset')
parser.add_argument('--random_views', type=bool,  default=True,
                    help='Render from random views')
parser.add_argument('--upper_views',  type=bool, default=True,
                    help='Render from only upper views')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--resolution', type=int, default=800,
                    help='How much camera deviates from the horizontal line')
parser.add_argument('--distance', type=float, default=4.0,
                    help='Distance between camera center and object')
parser.add_argument('--elevation', type=float, default=30.0,
                    help='How much camera deviates from the horizontal line')
parser.add_argument('--remove_doubles', type=bool, default=False,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--normalize', type=bool, default=True,
                    help='Normalize the object to (-1.0-1.0).')
parser.add_argument('--edge_split', type=bool, default=False,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)
print("===================================>random views {}".format(args.random_views))


# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
  links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
  # Remap as other types can not represent the full range of depth.
  map = tree.nodes.new(type="CompositorNodeMapValue")
  # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
  map.offset = [-0.7]
  map.size = [args.depth_scale]
  map.use_min = True
  map.min = [0]
  links.new(render_layers.outputs['Depth'], map.inputs[0])

  links.new(map.outputs[0], depth_file_output.inputs[0])

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = 'MULTIPLY'
# scale_normal.use_alpha = True
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = 'ADD'
# bias_normal.use_alpha = True
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

if args.normalize:
    load_obj(args.obj)
    file_name = os.path.basename(args.obj)
    output_path = args.obj.replace(file_name, "model.obj")
    print("export normalized obj to {}".format(output_path))
    bpy.ops.export_scene.obj(filepath=output_path)
else:
    bpy.ops.import_scene.obj(filepath=args.obj)

for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object
    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
        bpy.ops.object.transform_apply(scale=True)
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.shadow_method = 'NOSHADOW'
# Possibly disable specular shading:
lamp.use_specular = False
lamp.energy = lamp1_energy
print("lamp1:", lamp.type, lamp.energy)

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp2.shadow_method = 'NOSHADOW'
lamp2.use_specular = False
lamp2.energy = lamp2_energy
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180
print("lamp2:", lamp2.type, lamp2.energy)

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp3 = bpy.data.lamps['Sun.001']
lamp3.shadow_method = 'NOSHADOW'
lamp3.use_specular = False
lamp3.energy = lamp3_energy
bpy.data.objects['Sun.001'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun.001'].rotation_euler[0] += 90
print("lamp3:", lamp3.type, lamp3.energy)

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
print(bpy.data.lamps)
lamp4 = bpy.data.lamps['Sun.002']
lamp4.shadow_method = 'NOSHADOW'
lamp4.use_specular = False
lamp4.energy = lamp4_energy
bpy.data.objects['Sun.002'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun.002'].rotation_euler[0] += 270
print("lamp4:", lamp4.type, lamp4.energy)

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']
elevation = np.deg2rad(args.elevation)
cam.location = (0, args.distance, elevation)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = args.output_folder
scene.render.image_settings.file_format = 'PNG'  # set output format to .png



from math import radians

for output_node in [depth_file_output, normal_file_output, albedo_file_output]:
    output_node.base_path = ''
stepsize = 360.0 / args.views
rotation_mode = 'XYZ'
out_data['frames'] = []
for i in range(0, args.views):
    if args.random_views:
        scene.render.filepath = fp + '/r_' + str(i)
        if args.upper_views:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
        scene.render.filepath = fp + '/r_{0:03d}'.format(int(i * stepsize))

    if args.split=='test':
        scene.render.filepath = fp + '/r_' + str(i)
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

    bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': "r_{}".format(i),
        'rotation': radians(stepsize),
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

    if args.random_views:
        if args.upper_views:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        b_empty.rotation_euler[2] += radians(stepsize)


with open(fp + '/' + 'transforms.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)