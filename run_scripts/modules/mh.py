from . import api
from . import aws
from . import geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import time

def monostatic_iterated(pnts, object, cp, frequency):
    """The source points and field points are iterated over while the other data is fixed.
    Returns a matrix or vector of field point pressure values for the source points."""  

    p_reflect = []
    for pnt in pnts:
        api.set_initial_conditions(cp, frequency, 0.0)
        api.load_stl_mesh_to_cuda(object,0)        
        api.load_points_to_cuda([pnt], isSource=True)
        api.load_points_to_cuda([pnt], isSource=False)
        api.render_cuda()
        field_vals = api.GetFieldPoints(1)
        p_reflect.append([float(field_vals[0][0]), float(field_vals[0][1])])
        print("<<<<<<<<<-------------------->>>>")
        api.TearDownCuda()

    return p_reflect

def results_to_TES(res, tartget_range):
    """Convert the results to Target Strength."""
    # Convert the results to numpy array
    res = np.array(res)
    # Calculate the magnitudes
    magnitudes = np.sqrt(res[:, 0]**2 + res[:, 1]**2)
    atten = 20*np.log10(tartget_range)
    db_values = 20 * np.log10(magnitudes + 1e-12) + 2 * atten

    return db_values

def run_rendering(number_of_reflections = 0, project_target_to_feild_surface = True, test = False):
    """Wrapper function to run the rendering.
    Provides for a centralized place to call the rendering functions."""
    
    if test:
        project_source_to_feild_surface = True
        number_of_reflections = 0
        project_target_to_feild_surface = False
    else:
        project_source_to_feild_surface = False
    
    api.render_cuda()
    st_time = time.time()
    api.project_source_points_to_objects(project_source_to_feild_surface)
    end_time = time.time()
    print("Time taken to project source points to objects: ", end_time - st_time)
    
    for i in range(number_of_reflections):
        st_time = time.time()
        api.project_target_to_target_objects()
        end_time = time.time()
        print("Time taken to project target to target objects: ", end_time - st_time)
        
    if project_target_to_feild_surface:
        st_time = time.time()
        api.project_target_to_field_objects()
        end_time = time.time()
        print("Time taken to project target to field objects: ", end_time - st_time)
        
def render_to_file(viewSettings, window_width = 800*8, window_height = 600 * 8, file_name='output', test=False):
    """
    Render the scene to a file.
    The scene can take a long time to render so the resolution is set very high.
    The file name is timestamped to avoid overwriting previous files.
    """
    if test:
        file_name += "_test"
        window_width = 800
        window_height = 600
    api.render_openGL(window_width, window_height, viewSettings, file_name)

    print("Rendering to file: ", file_name)
    if not test:
        aws.copy_file_to_s3(file_name)
        # aws.copy_file_to_s3(file_name)
        # aws.list_as_url()
        
def setup_iso_scene(ref_length=20.0):
    """
    Sets up an isometric view for a render.
    Creates a field surface that is efficient for the range.
    """
    field_surface = geo.make_rectangle(ref_length,ref_length, False)

    edge_rhs = geo.make_right_angle_triangle(0.7*ref_length,0.31*ref_length)
    edge_rhs = geo.rotate_stl_object(edge_rhs, 'x', -90)
    edge_rhs = geo.translate_stl_object(edge_rhs, [-ref_length/2, 0, -ref_length/2])
    field_surface = geo.join_meshes(field_surface, edge_rhs)

    edge_rhs = geo.make_right_angle_triangle(0.31*ref_length,0.7*ref_length)
    edge_rhs = geo.rotate_stl_object(edge_rhs, 'x', -90)
    edge_rhs = geo.rotate_stl_object(edge_rhs, 'y', -90)
    edge_rhs = geo.translate_stl_object(edge_rhs, [-ref_length/2, 0, ref_length/2])
    field_surface = geo.join_meshes(field_surface, edge_rhs) 
             
    geo.rotate_stl_object(field_surface, 'y', -45)
    
    
    radius = ref_length * 12/20
    # Camera Postion
    viewSettings = [0.0] * 9
    viewSettings[0] = radius
    viewSettings[1] = radius
    viewSettings[2] = radius

    # Camera Target
    viewSettings[3] = 0.0
    viewSettings[4] = 0.0
    viewSettings[5] = 0.0

    # Camera Up Vector
    viewSettings[6] = 0.0
    viewSettings[7] = 1.0
    viewSettings[8] = 0.0
    
    return field_surface, viewSettings        



def seawater_attenuation_db_per_m(frequency_khz, temperature=10, salinity=35, depth=1000):
    """
    Estimate the attenuation of sound in seawater in dB/m for a given frequency (kHz).
    Uses a simplified Francois-Garrison formula for typical ocean conditions.
    Parameters:
        frequency_khz (float): Frequency in kHz
        temperature (float): Temperature in Celsius (default 10°C)
        salinity (float): Salinity in ppt (default 35)
        depth (float): Depth in meters (default 1000)
    Returns:
        float: Attenuation in dB/m
    """
    f = frequency_khz
    # Boric acid contribution (low frequency)
    A1 = 0.106
    f1 = 0.78 * np.sqrt(salinity/35) * np.exp(temperature/26)
    boric = (A1 * f1 * f**2) / (f1**2 + f**2)
    # Magnesium sulfate contribution (mid frequency)
    A2 = 0.52 * (1 + temperature/43) * (salinity/35)
    f2 = 42 * np.exp(temperature/17)
    mgso4 = (A2 * f2 * f**2) / (f2**2 + f**2)
    # Pure water contribution (high frequency)
    A3 = 0.00049
    pure_water = A3 * f**2
    # Total attenuation in dB/km
    attenuation_db_per_km = boric + mgso4 + pure_water
    # Convert to dB/m
    attenuation_db_per_m = attenuation_db_per_km / 1000.0
    return attenuation_db_per_m