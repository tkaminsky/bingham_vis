import numpy as np
import pyvista as pv
from deep_bingham.bingham_distribution import BinghamDistribution

dir = "meshes_new/"

name = "Arrow_cube"

path = dir + name + ".ply"

mesh_bottle = pv.read(dir + name + "_bottle.ply")
mesh_cap = pv.read(dir + name + "_bottle.ply")


# Create a Bingham distribution with a given M and Z
M_1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
M_2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
Z_1 = np.array([-10000,-100,-100,0])
Z_2 = np.array([-10000,-100,-100,0])

bingham_1 = BinghamDistribution(M_1, Z_1)
bingham_2 = BinghamDistribution(M_2, Z_2)

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

# Collect a sample of n points from the Bingham distribution
n = 100
sample_bottle = bingham_1.random_samples(n)
sample_cap = bingham_2.random_samples(n)

# Find the rotation matrix for each quaternion in the sample
axes_bottle = np.zeros((n,3))
angles_bottle = np.zeros(n)
axes_cap = np.zeros((n,3))
angles_cap = np.zeros(n)
for i in range(n):
    axis, angle = quaternion_axis_angle(sample_bottle[i])
    axes_bottle[i] = axis
    angles_bottle[i] = angle
    
    axis, angle = quaternion_axis_angle(sample_cap[i])
    axes_cap[i] = axis
    angles_cap[i] = angle
    
# Turn angles to degrees
angles_bottle = np.rad2deg(angles_bottle)
angles_cap = np.rad2deg(angles_cap)

# Center the mesh
mesh_bottle.points -= mesh_bottle.center
mesh_cap.points -= mesh_cap.center
# Scale the mesh

# mesh_bottle.scale(2, inplace=True)
# mesh_cap.scale(2, inplace=True)
    

# Plot the bunny meshes

plotter = pv.Plotter(shape=(2, 2))


# Copy the bottle mesh and rotate it by the predicted orientation
mean_axis, mean_angle = quaternion_axis_angle(M_1[:,2])
mesh_bottle_mean = mesh_bottle.copy()
mesh_bottle_mean.rotate_vector(vector=mean_axis, angle=mean_angle, inplace=True)
plotter.subplot(0, 0)
plotter.camera_position = 'xy'
plotter.camera.roll = 180
plotter.camera.azimuth = 30
plotter.camera.elevation = 30
plotter.add_text("Bottle Prediction", font_size=30)
plotter.add_mesh(mesh_bottle_mean, show_edges=True)

mean_axis, mean_angle = quaternion_axis_angle(M_2[:,2])
mesh_cap_mean = mesh_cap.copy()
mesh_cap_mean.rotate_vector(vector=mean_axis, angle=mean_angle, inplace=True)
plotter.subplot(0, 1)
plotter.camera_position = 'xy'
plotter.camera.roll = 180
plotter.camera.azimuth = 30
plotter.camera.elevation = 30
plotter.add_text("Cap Prediction", font_size=30)
plotter.add_mesh(mesh_cap_mean, show_edges=True)


# Create a new mesh for each point in the sample
obj_meshes_bottle = []
obj_meshes_cap = []
for i in range(n):
    # Rotate the bunny mesh by the given axis and angle
    bunny = mesh_bottle.copy()
    bunny.rotate_vector(vector=axes_bottle[i], angle=angles_bottle[i], inplace=True)
    obj_meshes_bottle.append(bunny)

    bunny = mesh_cap.copy()
    bunny.rotate_vector(vector=axes_cap[i], angle=angles_cap[i], inplace=True)
    obj_meshes_cap.append(bunny)



plotter.subplot(1,0)
plotter.camera_position = 'xy'
plotter.camera.roll = 180
plotter.camera.azimuth = 30
plotter.camera.elevation = 30
plotter.add_text("Bottle Uncertainty", font_size=30)
for i in range(n):
    plotter.add_mesh(obj_meshes_bottle[i], show_edges=True)
plotter.subplot(1,1)
plotter.camera_position = 'xy'
plotter.camera.roll = 180
plotter.camera.azimuth = 30
plotter.camera.elevation = 30
plotter.add_text("Cap Uncertainty", font_size=30)
for i in range(n):
    plotter.add_mesh(obj_meshes_cap[i], show_edges=True)
    
plotter.show()