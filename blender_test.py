import bpy

camera_data = bpy.data.cameras.new("Camera")
camera_object = bpy.data.objects.new(name="Camera", object_data=camera_data)

bpy.ops.wm.open_mainfile(filepath="FilesForKu/Arrow_cube_bottle.blend")

scene = bpy.context.scene
scene.collection.objects.link(camera_object)

camera_object.location = (0, 0, 10)
camera_object.rotation_euler = (0, 0, 0)

render = scene.render
render.use_overwrite = False
render.use_file_extension = True
render.filepath = "output.png"
render.resolution_x = 800
render.resolution_y = 800

bpy.ops.render.render(write_still=True)

bpy.ops.image.open(filepath="output.png")