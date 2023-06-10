## Import all relevant libraries
import bpy
import numpy as np
import math as m
import random
import os
from PIL import Image

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import h5py

def load_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn, 'NodeTree', name)
    bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
  """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  # Figure out how many materials are already in the scene
  mat_count = len(bpy.data.materials)

  # Create a new material; it is not attached to anything and
  # it will be called "Material"
  bpy.ops.material.new()

  # Get a reference to the material we just created and rename it;
  # then the next time we make a new material it will still be called
  # "Material" and we will still be able to look it up by name
  mat = bpy.data.materials['Material']
  mat.name = 'Material_%d' % mat_count

  # Attach the new material to the active object
  # Make sure it doesn't already have materials
  obj = bpy.context.active_object
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # Find the output node of the new material
  output_node = None
  for n in mat.node_tree.nodes:
    if n.name == 'Material Output':
      output_node = n
      break

  # Add a new GroupNode to the node tree of the active material,
  # and copy the node tree from the preloaded node group to the
  # new group node. This copying seems to happen by-value, so
  # we can create multiple materials of the same type without them
  # clobbering each other
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
  )

colors = np.array([
    [255, 127, 80, 255],  # Coral
    [0, 255, 127, 255],   # Spring Green
    [127, 255, 0, 255],   # Chartreuse
    [255, 215, 0, 255],   # Gold
    [255, 69, 0, 255],    # Orange-Red
    [72, 209, 204, 255],  # Medium Turquoise
    [255, 165, 0, 255],   # Orange
    [154, 205, 50, 255],  # Yellow Green
    [65, 105, 225, 255],  # Royal Blue
    [218, 112, 214, 255], # Orchid
    [255, 140, 0, 255],   # Dark Orange
    [32, 178, 170, 255],  # Light Sea Green
    [135, 206, 250, 255], # Light Sky Blue
    [255, 192, 203, 255], # Pink
    [0, 128, 128, 255],   # Teal
    [186, 85, 211, 255],  # Medium Orchid
    [255, 99, 71, 255],   # Tomato
    [0, 191, 255, 255],   # Deep Sky Blue
    [250, 128, 114, 255], # Salmon
    [147, 112, 219, 255], # Medium Purple
], dtype=np.uint8)

colors = np.array([
   [217, 217,217, 255],
    [255, 0, 0, 255],      # Red
    [0, 255, 0, 255],      # Green
    [0, 0, 255, 255],      # Blue
    [255, 255, 0, 255],    # Yellow
    [255, 0, 255, 255],    # Magenta
    [0, 255, 255, 255],    # Cyan
    [255, 128, 0, 255],    # Orange
    [128, 0, 255, 255],    # Purple
    [0, 255, 128, 255],    # Lime
    [128, 255, 0, 255],    # Chartreuse
    [255, 0, 128, 255],    # Pink
    [0, 128, 255, 255],    # Sky Blue
    [255, 128, 128, 255],  # Light Pink
    [128, 255, 128, 255],  # Light Green
    [128, 128, 255, 255],  # Light Blue
    [123, 87, 50, 255],    # Brown
    [70, 130, 180, 255],   # Steel Blue
    [255, 192, 203, 255],  # Pink
    [0, 128, 0, 255],      # Forest Green
    [255, 255, 128, 255]   # Light Yellow
], dtype=np.uint8)


colors = colors // 4

# Normalize the color values to the range [0, 1]
colors = colors.astype(np.float32) / 255.0

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])

def set_scene(filepath):
    bpy.ops.wm.open_mainfile(filepath=filepath)
    
    load_materials('./mats')

    # Add a camera to the scene
    camera_data = bpy.data.cameras.new("Camera")
    camera_object = bpy.data.objects.new(name="Camera", object_data=camera_data)
    scene = bpy.context.scene
    scene.collection.objects.link(camera_object)

    # Make the camera the active camera
    scene.camera = camera_object

    # Add an axis and place the camera under it
    axis = bpy.data.objects.new(name="Main Axis", object_data=None)
    scene.collection.objects.link(axis)
    axis.rotation_euler = (0, 0, 0)
    axis.location = (0, 0, .25)
    camera_object.parent = axis
    
    bpy.context.view_layer.objects.active = bpy.context.scene.objects[2]
    bpy.context.object.data.materials.clear()
    add_material("Rubber")

    bpy.data.materials["Material_3"].node_tree.nodes["Group"].inputs[0].default_value = colors[0] #(217, 217, 217, 255)
    
    
    #bpy.data.materials["Red"].use_nodes = False
    
    # base_light_count = 1
    # norm = .3
    # bl_positions = [
    #     [.5,0,0],
    #     [-.5,0,0],
    #     [0,.5,0],
    #    [0,-.5,0],
    #    [0,0,.5],
    #    [.25, .25, 0],
    #    [.25, -.25, 0],
    #    [-.25, .25, 0],
    #    [-.25,-.25, 0]
    # ]
    
    # bl_positions = [
    #   [0,0,1],
    #   [1,0,0],
    #   [-1/2, np.sqrt(3)/2,0],
    #   [-1/2,-np.sqrt(3)/2, 0]
    # ]
    
    
    # base_light_count = len(bl_positions)
    # print(len(bl_positions))
    
    # for i in range(base_light_count):
    #     light_data = bpy.data.lights.new(name=f"Light{i}", type='POINT')
    #     light_object = bpy.data.objects.new(name=f"Light{i}", object_data=light_data)
    #     scene.collection.objects.link(light_object)
    #     bl_positions[i] = bl_positions[i] / np.linalg.norm(bl_positions[i]) * norm
    #     bl_positions[i][2] += .2
    #     light_object.location = (bl_positions[i][0], bl_positions[i][1], bl_positions[i][2])
    #     light_object.data.energy = 20
        
    light_data = bpy.data.lights.new(name="RoamingLight", type='POINT')
    light_object = bpy.data.objects.new(name="RoamingLight", object_data=light_data)
    scene.collection.objects.link(light_object)
    light_object.location = (0, 0, 1)


    return scene, axis, light_object

def change_color(color):
    bpy.data.materials["Material_3"].node_tree.nodes["Group"].inputs[0].default_value = colors[0]
    #bpy.ops.wm.save_as_mainfile(filepath="/home/tkaminsky/Desktop/bingham/bingham_vis/test.blend")

    # bpy.data.materials["Red"].metallic = 0
    # bpy.data.materials["Blue"].metallic = 0
    # bpy.data.materials["Blue"].roughness = .55
    # bpy.data.materials["Red"].roughness = .55
    # bpy.data.materials["Blue"].diffuse_color = color
    # bpy.data.materials["Red"].diffuse_color = color
    # bpy.data.materials["Blue"].specular_intensity = .5
    # bpy.data.materials["Red"].specular_intensity = .5
setting = "Full"


randomize_color = True
randomize_light = True

if setting == "Default":
  dir = "FilesForKu/"
  objects = ['Arrow_cube', 'Circle_cube', 'Cross_cube', 'Diamond_cube', 'Hexagon_cube', 'Key_cube', 'Line_cube', 'Pentagon_cube', 'U_cube']
  #objects = ['Line_cube', 'Pentagon_cube', 'U_cube']
  MIN = -m.pi / 3
  MAX = m.pi / 3
  STEP = m.pi / 18

  # Randomly sample H orientations
  H = 500
  x_rots = np.random.uniform(MIN, MAX, H)
  y_rots = np.random.uniform(MIN, MAX, H)
  z_rots = np.random.uniform(0, 2 * m.pi, H)

  # Convert to scipy rotations
  sample = np.zeros((H, 3, 3))
  for i in range(H):
      sample[i] = R.from_euler('XYZ', [x_rots[i], y_rots[i], z_rots[i]]).as_matrix()

elif setting == "Simple":
  dir = "FilesForKu/"
  objects = ['Cross_cube']
  #objects = ['Line_cube', 'Pentagon_cube', 'U_cube']
  MIN = -m.pi / 3
  MAX = m.pi / 3
  STEP = m.pi / 18

  # Randomly sample H orientations
  H = 500
  x_rots = np.random.uniform(MIN, MAX, H)
  y_rots = np.random.uniform(MIN, MAX, H)
  z_rots = np.random.uniform(0, 2 * m.pi, H)

  # Convert to scipy rotations
  sample = np.zeros((4 * H, 3, 3))
  for i in range(H):
      for j in range(4):
        # Rotate by multiples of 90 degrees
        sample[4 * i + j] = R.from_euler('XYZ', [x_rots[i], y_rots[i], z_rots[i] + j * m.pi / 2]).as_matrix()
        #sample[i] = R.from_euler('xyz', [x_rots[i], y_rots[i], z_rots[i]]).as_matrix()

elif setting == "Full":
  dir = "FilesForKu/"
  objects = ['Arrow_cube', 'Circle_cube', 'Cross_cube', 'Diamond_cube', 'Hexagon_cube', 'Key_cube', 'Line_cube', 'Pentagon_cube', 'U_cube']
  MIN = -m.pi / 3
  MAX = m.pi / 3
  STEP = m.pi / 18

  # Randomly sample H orientations
  H = 500
  x_rots = np.random.uniform(MIN, MAX, H)
  y_rots = np.random.uniform(MIN, MAX, H)
  z_rots = np.random.uniform(0, 2 * m.pi, H)

  # Convert to scipy rotations
  sample = np.zeros((4 * H, 3, 3))
  for i in range(H):
      for j in range(4):
        # Rotate by multiples of 90 degrees
        sample[4 * i + j] = R.from_euler('XYZ', [x_rots[i], y_rots[i], z_rots[i] + j * m.pi / 2]).as_matrix()
        #sample[i] = R.from_euler('xyz', [x_rots[i], y_rots[i], z_rots[i]]).as_matrix()
  
  randomize_color = False
  randomize_light = False
   

def generate_random_data(object, name='bc_data'):
  files = [f'{dir}{object}_bottle.blend', f'{dir}{object}_cap.blend']
  types = ['Bottle', 'Cap']



  # Create a new hdf5 file
  f = h5py.File(f'{object}_data.hdf5', 'w')
  g_cap = f.create_group('Cap')
  g_bottle = f.create_group('Bottle')

  # Create a dataset of images and a dataset of orientations in each group
  dset_cap = g_cap.create_dataset('images', (0, 256, 256, 3), maxshape=(None, 256, 256, 3), dtype='uint8')
  dset_bottle = g_bottle.create_dataset('images', (0, 256, 256, 3), maxshape=(None, 256, 256, 3), dtype='uint8')

  dset_cap_angles = g_cap.create_dataset('angles', (0, 3, 3), maxshape=(None, 3, 3), dtype='float32')
  dset_bottle_angles = g_bottle.create_dataset('angles', (0, 3, 3), maxshape=(None, 3, 3), dtype='float32')

  # if not os.path.exists(f'{name}/{object}'):
  #         os.makedirs(f'{name}/{object}')
  #         os.makedirs(f'{name}/{object}/Bottle')
  #         os.makedirs(f'{name}/{object}/Cap')

  for i in range(len(files)):
      # Set the scene
      scene, axis, light_object = set_scene(files[i])
      energy_bounds = [50, 150]
      light_object.location = (0,0,.5)
      light_object.data.energy = 50

      # Make the background black
      world = bpy.data.worlds['World']
      world.use_nodes = True
      bg_node = world.node_tree.nodes['Background']
      bg_node.inputs[0].default_value[:3] = (0, 0, 0)

      # Get the object
      obj_now = scene.objects[types[i]]

      max_n = len(sample)

      for s in range(len(sample)):
          # light_pos = (np.random.rand(2) - 0.5) 
          # # Change the light position to a random position
          # light_object.location = (light_pos[0], light_pos[1], .5)
          # # Change energy 
          # light_object.data.energy = np.random.uniform(energy_bounds[0], energy_bounds[1])
          #light_object.data.energy = 50

          # Randomize the color
          #change_color(random.choice(colors))

          if s % 10 == 0:
              print(f'{s}/{max_n}')

          x, y, z = R.from_matrix(sample[s]).as_euler('xyz')
          #x, y, z = x_rots[s], y_rots[s], z_rots[s]
          obj_now.rotation_euler = (x, y, z)
          render = scene.render
          render.use_overwrite = False
          render.use_file_extension = True

          #save_name = f"{name}/{object}/{types[i]}/{'{:08d}'.format(s)}.png"
          save_name = "test.png"

          render.resolution_x = 256
          render.resolution_y = 256
          render.resolution_percentage = 100
          render.filepath = save_name

          bpy.ops.render.render(write_still=True)

          # Load the image as a PIL image
          img_pil = Image.open(save_name)
          img_pil = img_pil.resize((256, 256))
          img_pil = img_pil.convert('RGB')
          img = np.array(img_pil)


          # Add the image to the dataset
          if types[i] == 'Cap':
              dset_cap.resize(dset_cap.shape[0] + 1, axis=0)
              dset_cap[-1:] = img
              dset_cap_angles.resize(dset_cap_angles.shape[0] + 1, axis=0)
              dset_cap_angles[-1:] = sample[s]
          else:
              dset_bottle.resize(dset_bottle.shape[0] + 1, axis=0)
              # Print the shape of the dataset

              dset_bottle[-1:] = img
              dset_bottle_angles.resize(dset_bottle_angles.shape[0] + 1, axis=0)
              dset_bottle_angles[-1:] = sample[s]


# def generate_data(object, randomize_color=False):
#     files = [f'{dir}{object}_bottle.blend', f'{dir}{object}_cap.blend']
#     types = ['Bottle', 'Cap']

#     if not os.path.exists(f'bc_data/{object}'):
#             os.makedirs(f'bc_data/{object}')
#             os.makedirs(f'bc_data/{object}/Bottle')
#             os.makedirs(f'bc_data/{object}/Cap')

#     for i in range(len(files)):
#         scene, axis, light_object = set_scene(files[i])
#         energy_bounds = [50, 150]
         
#         gt_angles = np.zeros((len(angles) * len(angles_second) * len(angles_third), 4))

#         obj_now = scene.objects[types[i]]
        
#         n = 0
#         idx = 0

#         for j in range(len(angles)):
#             x = angles[j]
#             for k in range(len(angles_second)):
#                 y = angles_second[k]
#                 for l in range(len(angles_third)):
#                     z = angles_third[l]

#                     light_pos = (np.random.rand(2) - 0.5) 
#                     # Change the light position to a random position
#                     light_object.location = (light_pos[0], light_pos[1], .5)
#                     # Change energy 
#                     light_object.data.energy = np.random.uniform(energy_bounds[0], energy_bounds[1])
#                     light_object.data.energy = 50


#                     change_color(random.choice(colors))

#                     if n % 10 == 0:
#                         print(f'{n}/{max_n}')

#                     # axis.rotation_euler = (x, y, z)
#                     obj_now.rotation_euler = (x, y, z)
#                     render = scene.render
#                     render.use_overwrite = False
#                     render.use_file_extension = True

#                     save_name = f"bc_data/{object}/{types[i]}/{'{:08d}'.format(n)}.png"
#                     #save_name = "test.png"

#                     render.filepath = save_name
#                     render.resolution_x = 600
#                     render.resolution_y = 600

#                     #gt_angle_euler = np.array([x,y,z])
#                     # Turn the euler angles into a quaternion
#                     gt_angle_quat = get_quaternion_from_euler(x, y, z)

#                     gt_angles[idx] = gt_angle_quat
#                     idx += 1

#                     bpy.ops.render.render(write_still=True)

#                     # Write a blurb about what line 86 does 
#                     # This is the line that saves the image
#                     bpy.ops.image.open(filepath=save_name)
#                     n += 1

#         np.save(f'bc_data/{object}/{types[i]}/angles.npy', gt_angles)


object_list = objects

for obj_atm in object_list:
    print(f"Generating data for {obj_atm}")
    generate_random_data(obj_atm, name="bc_data_mat")

print("Done.")