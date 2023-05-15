import bpy
import numpy as np
import pyvista as pv
# from bpy.app.handlers import persistent

# @persistent
# def load_handler(dummy):
#     print("Load Handler:", bpy.data.filepath)

# bpy.app.handlers.load_post.append(load_handler)

dir = "FilesForKu/"

file = "U_cube_bottle"

path = dir + file + ".blend"

bpy.ops.wm.open_mainfile(filepath=path)

bpy.ops.export_mesh.ply(filepath=f"meshes/{file}.ply")