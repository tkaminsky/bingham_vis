import numpy as np
import bpy
from deep_bingham.bingham_distribution import BinghamDistribution

dir = "meshes/"

name = "Arrow_cube"

bottle_file = dir + name + "_bottle.ply"
cap_file = dir + name + "_cap.ply"

bottle = None
cap = None

# Perform a quaternion multiplication
def q_mult(a,b):
    a0,a1,a2,a3 = a
    b0,b1,b2,b3 = b
    
    c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3
    c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2
    c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1
    c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0
    
    return np.array([c0,c1,c2,c3])

# Negate a quaternion
def q_neg(a):
    a0,a1,a2,a3 = a
    return np.array([a0,-a1,-a2,-a3])

# Rotate a vector by a quaternion
def rotate(v, q):
    return q_mult(q_mult(q_neg(q), v), q)

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def quaternion_axis_angle(q):
    # Turn the quaternion into an axis-angle representation
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    
    # Extract the axis
    axis = np.array([q1,q2,q3])
    axis = axis / np.linalg.norm(axis)
    
    # Extract the angle
    angle = 2 * np.arccos(q0)
    
    return axis, angle


N = 100

I = np.eye(3) / 100

class BinghamObject():
    def __init__(self, obj, M, Z):
        self.obj = obj
        self.M = M
        self.Z = Z
        self.bingham = BinghamDistribution(M, Z)
        self.points = self.get_points()
        self.x = self.points[:,0,:] # x-coordinates
        self.y = self.points[:,1,:] # y-coordinates
        self.z = self.points[:,2,:] # z-coordinates

    def get_points(self):
        points = np.zeros((N, 3, 3))
        sample = self.bingham.random_samples(N)
        for i in range(N):
            # Rotate the standard axes (I) by the sample quaternion
            R = quaternion_rotation_matrix(sample[i])

            points[i] = np.matmul(R, I)
            
        return points

# Create a new scene
scene = bpy.data.scenes.new("Scene")

# Set the new scene as the active scene
bpy.context.window.scene = scene

# Load the bottle file
bpy.ops.import_mesh.ply(filepath=bottle_file)
bottle = bpy.context.selected_objects[0]

# Load the cap file
# bpy.ops.import_mesh.ply(filepath=cap_file)
# cap = bpy.context.selected_objects[0]


# Position the object in the new scene as desired
# For example:
bottle.location = (.25, .25, 0)
# cap.location = (-.25, .25, 0)

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
axis.location = (0, 0, .5)
camera_object.parent = axis

# Add a light to the scene
light_data = bpy.data.lights.new(name="Light", type='AREA')
light_object = bpy.data.objects.new(name="Light", object_data=light_data)
# Change the energy of the light to make it brighter
light_object.data.energy = 100

scene.collection.objects.link(light_object)
light_object.location = (0, 0, 1)

# # Create a Bingham distribution with a given M and Z
# M_1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# M_2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# Z_1 = np.array([-10000,-100,-100,0])
# Z_2 = np.array([-10000,-100,-100,0])

# bottle_dist = BinghamObject(bottle, M_1, Z_1)
# cap_dist = BinghamObject(cap, M_2, Z_2)

# def add_point_cloud(points, name, color=(.1,.1,.1, 1)):
#     # Create a new mesh
#     mesh = bpy.data.meshes.new(name)
#     mesh.from_pydata(points, [], [])

#     # Create a new object and link it to the scene
#     obj = bpy.data.objects.new(name, mesh)
#     bpy.context.scene.collection.objects.link(obj)

#     # Set the object to be displayed as a point cloud
#     obj.display_type = 'SOLID'

#     # new_mat = bpy.data.materials.new(name=name)
#     # print(new_mat.diffuse_color[1])
#     # print(color)
#     # new_mat.diffuse_color = color
#     # obj.material_slots.material = new_mat

#     return obj

# # Add the point clouds to the scene
# bottle_cloud_x = add_point_cloud(bottle_dist.x, "Bottle Cloud X", color=(1,0,0,1))
# # Make the bottle cloud red

# bottle_cloud_y = add_point_cloud(bottle_dist.y, "Bottle Cloud Y", color=(0,1,0,1))
# # Make the bottle cloud green

# bottle_cloud_z = add_point_cloud(bottle_dist.z, "Bottle Cloud Z", color=(0,0,1,1))
# # Make the bottle cloud blue


# cap_cloud_x = add_point_cloud(cap_dist.x, "Cap Cloud X", color=(1,0,0, 1))

# cap_cloud_y = add_point_cloud(cap_dist.y, "Cap Cloud Y", color=(0,1,0, 1))

# cap_cloud_z = add_point_cloud(cap_dist.z, "Cap Cloud Z", color=(0,0,1, 1))


# # Move the point clouds so the points are all relative to (.25, -.25, 0)
# bottle_cloud_x.location = (-.25, .25, 0)
# bottle_cloud_y.location = (-.25, .25, 0)
# bottle_cloud_z.location = (-.25, .25, 0)
# cap_cloud_x.location = (-.25, -.25, 0)
# cap_cloud_y.location = (-.25, -.25, 0)
# cap_cloud_z.location = (-.25, -.25, 0)

# # Add a translucent sphere centered at both of (-.25, .25, 0) and (-.25, -.25, 0)
# bpy.ops.mesh.primitive_uv_sphere_add(location=(-.25, .25, 0), radius=.01)
# bpy.ops.mesh.primitive_uv_sphere_add(location=(-.25, -.25, 0), radius=.01)

# Render the scene
render = scene.render
render.use_overwrite = False
render.use_file_extension = True
render.filepath = f"output.png"
render.resolution_x = 800
render.resolution_y = 800

bpy.ops.render.render(write_still=True)

# Write a blurb about what line 86 does 
# This is the line that saves the image
bpy.ops.image.open(filepath=f"output.png")

    
