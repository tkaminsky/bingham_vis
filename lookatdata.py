import h5py
import numpy as np
import os

db = h5py.File('bc_data.hdf5', 'r')

item = db['Hexagon_cube']['bottle']['images'][0]
label = db['Hexagon_cube']['bottle']['labels'][0]

# Print type and shape of each
print("Shapes and types of item and label:")
print(type(item))
print(item.shape)
print(type(label))
print(label.shape)
print("END\n\n")

# Visualize the image
import matplotlib.pyplot as plt

print("Pose:" + str(label))

print("Item:")
plt.imshow(item)
plt.show()
print("End item\n\n")

print(db.keys())

for k in db.keys():
    print(db[k].keys())
    for j in db[k].keys():
        print(db[k][j].keys())
        print("Shape of images and labels:")
        print(db[k][j]['images'].shape)
        print(db[k][j]['labels'].shape)
        print("END\n\n")