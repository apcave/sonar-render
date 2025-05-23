import TES_Model_API as api

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

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