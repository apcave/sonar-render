import modules.api as api
import modules.mh as mh
import modules.geo as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math




def point_on_xz_plane(angle_deg, radius):
    """
    Returns (x, z) coordinates for a point in the xz-plane,
    given an angle in degrees from the +z axis and a radius.
    """
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.sin(angle_rad)
    z = radius * np.cos(angle_rad)
    print(f"Point on xz-plane at angle {angle_deg} degrees: x = {x}, z = {z}")
    return [[x, 0, z]]




def render_cube(frequency, angle_i, r_rotation):

    test = False
    cp = 1480.0

    target = geo.create_reflector(2, 0.01)
    #target.vectors = target.vectors * 2  # Scale down the mesh to 1 cm size
    #target = geo.rotate_stl_object(target, 'x', -20)
    #target = geo.rotate_stl_object(target, 'y', 10)
    y_offset = 0
    target = geo.translate_stl_object(target, [0, y_offset, -5])
    target = geo.rotate_stl_object(target, 'z', r_rotation)

    for i in range(4):
        target = geo.halve_facets(target)

    #15
    # ref_length = 25
    ref_length = 10
    field_surface, viewSettings,  = mh.setup_iso_scene(ref_length)
    
    # Sometimes smaller facets are required as very large facets do not render.
    for i in range(4):
        field_surface = geo.halve_facets(field_surface)    

    # for i in range(3):
    #     field_surface = geo.halve_facets(field_surface)

    t = 1000

    source_pnts = point_on_xz_plane(angle_i, t)


    delta_s = (cp / frequency) / 7.5
    delta_s = min(delta_s, 50e-3)  # Ensure a minimum distance for source points.
    print("delta_s = ", delta_s)

    delta_t = (cp / frequency) / 7.5
    #delta_t = min(delta_t, 50e-3)  # Ensure a minimum distance for source points.
    print("delta_t = ", delta_t)

    api.load_points_to_cuda(source_pnts, isSource=True)
    api.set_initial_conditions(cp, frequency, 0.0)
    api.load_stl_mesh_to_cuda(target,0, delta_s) # 0 is for target object.
    api.load_stl_mesh_to_cuda(field_surface, 2, delta_t) # 1 is for field surface.

    #for frequency in range(1e3, 30e3, 500):
    
    # file_name = "range_test"
    api.set_initial_conditions(cp, frequency, 0.0)

    mh.run_rendering(10, test=test)
    
    file_name = f"reflector_angle_v2_{int(frequency)}_{int(angle_i)}_deg"
    api.sound_visualisation_init(-45,-110, True, True)
    mh.render_to_file(viewSettings, file_name=file_name, test=test)
    
    # file_name = f"reflector_angle_bw_{int(frequency)}_{int(angle_i)}_deg"
    # api.sound_visualisation_init(-30,-100, True, False)
    # mh.render_to_file(viewSettings, file_name=file_name, test=test)
    
    
    api.TearDownCuda()

mh.query_gpu_info()

#render_cube(10.0e3, 20, 0)

for angle in range(0, 76, 1):
    print(f"Rendering cube at angle: {angle} degrees")
    render_cube(10.0e3, angle, 0)

