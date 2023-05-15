import h5py
import numpy as np
import os
from PIL import Image

dir = 'bc_data'

files_list = []

# For each directory in bc_data
for object in os.listdir(dir):
    # Remove the .DS_Store file
    if object == '.DS_Store':
        continue
    # For each type of object (bottle or cap)
    for type in os.listdir(f'{dir}/{object}'):
        if type == '.DS_Store':
            continue
        files_list.append(f'{object}/{type}')

print(files_list)
