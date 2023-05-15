## Import all relevant libraries
import bpy
import numpy as np
import math as m
import random
import os

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

# Normalize the color values to the range [0, 1]
colors = colors.astype(np.float32) / 255.0

dir = "FilesForKu/"
objects = ['Arrow_cube', 'Circle_cube', 'Cross_cube', 'Diamond_cube', 'Hexagon_cube', 'Key_cube', 'Line_cube', 'Pentagon_cube', 'U_cube']

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

    # Add a light to the scene
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)
    scene.collection.objects.link(light_object)
    light_object.location = (0, 0, 1)

    return scene, axis, light_object

def change_color(color):
    bpy.data.materials["Red"].node_tree.nodes["Diffuse BSDF"].inputs[0].default_value = color
    bpy.data.materials["Blue"].node_tree.nodes["Diffuse BSDF"].inputs[0].default_value = color



MIN = -m.pi / 3
MAX = m.pi / 3
STEP = m.pi / 18
angles = np.arange(MIN, MAX, STEP)
#angles = np.array([0])
angles_second = angles
angles_third = np.arange(0, 2 * m.pi, m.pi / 10)
max_n = len(angles) * len(angles_second) * len(angles_third)

print(len(angles))
print(len(angles) ** 3)

def generate_data(object, randomize_color=False):
    files = [f'{dir}{object}_bottle.blend', f'{dir}{object}_cap.blend']
    types = ['Bottle', 'Cap']

    if not os.path.exists(f'bc_data/{object}'):
            os.makedirs(f'bc_data/{object}')
            os.makedirs(f'bc_data/{object}/bottle')
            os.makedirs(f'bc_data/{object}/cap')

    for i in range(len(files)):
        scene, axis, light_object = set_scene(files[i])
        energy_bounds = [light_object.data.energy * .5, light_object.data.energy * 1.5]
         
        gt_angles = np.zeros((len(angles), len(angles_second), len(angles_third), 4))

        obj_now = scene.objects[types[i]]
        
        n = 0

        for j in range(len(angles)):
            x = angles[j]
            for k in range(len(angles_second)):
                y = angles_second[k]
                for l in range(len(angles_third)):
                    z = angles_third[l]

                    light_pos = np.random.rand(2) - 0.5
                    # Change the light position to a random position
                    light_object.location = (light_pos[0], light_pos[1], 1)
                    # Change energy 
                    light_object.data.energy = np.random.uniform(energy_bounds[0], energy_bounds[1])

                    change_color(random.choice(colors))

                    if n % 10 == 0:
                        print(f'{n}/{max_n}')

                    # axis.rotation_euler = (x, y, z)
                    obj_now.rotation_euler = (x, y, z)
                    render = scene.render
                    render.use_overwrite = False
                    render.use_file_extension = True

                    save_name = f"bc_data/{object}/{types[i]}/{'{:08d}'.format(n)}.png"
                    #save_name = "test.png"

                    render.filepath = save_name
                    render.resolution_x = 600
                    render.resolution_y = 600

                    #gt_angle_euler = np.array([x,y,z])
                    # Turn the euler angles into a quaternion
                    gt_angle_quat = get_quaternion_from_euler(x, y, z)

                    gt_angles[j, k, l] = gt_angle_quat

                    bpy.ops.render.render(write_still=True)

                    # Write a blurb about what line 86 does 
                    # This is the line that saves the image
                    bpy.ops.image.open(filepath=save_name)
                    n += 1

        np.save(f'bc_data/{object}/{types[i]}/angles.npy', gt_angles)


object_list = objects

for obj_atm in object_list:
    print(f"Generating data for {obj_atm}")
    generate_data(obj_atm)

print("Done.")