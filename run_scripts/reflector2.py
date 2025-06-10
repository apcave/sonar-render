import modules.api as api
import modules.mh as mh
import modules.geo as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math









def render_cube(frequency):

    cp = 1480.0
    test = False
    cp = 1480.0

    target = geo.create_reflector(2, 0.01)
    #target.vectors = target.vectors * 2  # Scale down the mesh to 1 cm size
    #target = geo.rotate_stl_object(target, 'x', -20)
    #target = geo.rotate_stl_object(target, 'y', 10)
    target = geo.translate_stl_object(target, [0, 2, -5])
    #target = geo.rotate_stl_object(target, 'y', 35)

    for i in range(4):
        target = geo.halve_facets(target)

    #15
    ref_length = 25
    field_surface, viewSettings,  = mh.setup_iso_scene(ref_length)
    
    # Sometimes smaller facets are required as very large facets do not render.
    for i in range(3):
        field_surface = geo.halve_facets(field_surface)    

    # for i in range(3):
    #     field_surface = geo.halve_facets(field_surface)


    angle_i = [0]
    t = 20

    source_pnts=[[0*t,-1*t+5,9*t]]


    delta_s = (cp / frequency) / 7.5
    #delta_s = min(delta_s, 25e-3)  # Ensure a minimum distance for source points.
    print("delta = ", delta_s)

    delta_t = (cp / frequency) / 4
    #delta_t = min(delta_t, 50e-3)  # Ensure a minimum distance for source points.
    print("delta_t = ", delta_t)

    api.load_points_to_cuda(source_pnts, isSource=True)
    api.set_initial_conditions(cp, frequency, 0.0)
    api.load_stl_mesh_to_cuda(target,0, delta_s) # 0 is for target object.
    api.load_stl_mesh_to_cuda(field_surface, 2, delta_t) # 1 is for field surface.

    #for frequency in range(1e3, 30e3, 500):
    file_name = f"reflector2_test_{int(frequency)}_Hz"
    file_name = "range_test"
    api.set_initial_conditions(cp, frequency, 0.0)
    mh.run_rendering(1, test=test)

    mh.render_to_file(viewSettings, file_name=file_name, test=test)
    api.TearDownCuda()

mh.query_gpu_info()

render_cube(5.0e3)
# for frequency in range(1920, 100000, 1):
#     print(f"Rendering cube at frequency: {frequency} Hz")
#     render_cube(float(frequency))