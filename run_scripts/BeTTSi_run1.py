import modules.api as api
import modules.mh as mh
import modules.geo as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import datetime










cp = 1480.0

# 1e3 = 30 sec
# 2e3 = 120 sec
# 4e3 = 480 sec


target = geo.load_stl_file('../hull_20cmMesh.stl')
target = geo.rotate_stl_object(target, 'x', -90)
target = geo.rotate_stl_object(target, 'y', -90)
target = geo.translate_stl_object(target, [0, 5, 0])



def run_at_frequency(frequency):
    """Run the simulation at the given frequency."""

    test = False

    estimate_time = 30*(2**(frequency / 1000))
    print(f"Starting simulation at {frequency} Hz, estimated time: {estimate_time} sec")
    print("Start time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Estimated end time: ", (datetime.datetime.now() + datetime.timedelta(seconds=estimate_time)).strftime("%Y-%m-%d %H:%M:%S"))

    file_name = f"BeTSSi_R1_{int(frequency)}_Hz"


    ref_length = 100
    # 1000, 439 sec
    # 500, 105 sec
    approx_number_of_field_fragments_per_side = 300 

    field_surface, viewSettings,  = mh.setup_iso_scene(ref_length)


    for i in range(3):
        field_surface = geo.halve_facets(field_surface)

    #145.39013504981995

    angle_i = [0]
    t = 1000
    #source_pnts=[[0*4*t,0*3*t,9*t],[0,3*t,-9*t]]
    source_pnts=[[1*t,100,0.5*t]]
    angles = np.linspace(-180, 180, 361, endpoint=False)

    delta = (cp / frequency) / 7.5
    delta_s = min(delta, 50e-3)  # Ensure a minimum distance for source points.
    print("delta = ", delta_s)
    delta_f = ref_length / approx_number_of_field_fragments_per_side
    delta_f = (cp / frequency) / 7.5
    att = mh.seawater_attenuation_db_per_m(frequency/1000)

    api.load_points_to_cuda(source_pnts, isSource=True)
    api.set_initial_conditions(cp, frequency, att)
    api.sound_visualisation_init(-40,-100, True, True)

    api.load_stl_mesh_to_cuda(target, 0, delta_s) # 0 is for target object.
    api.load_stl_mesh_to_cuda(field_surface, 2, delta_f) # 1 is for field surface.

    mh.run_rendering(0, test=test)


    mh.render_to_file(viewSettings, file_name=file_name, test=test)
    api.TearDownCuda()


frequency = 0.0
while True:
    frequency += 1000.0
    run_at_frequency(frequency)