import TES_Model_API as api
import ModelHelpers as mh
import Geometry as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

def mono_rectangle_tes(a,b,wavelength, target_range,angles = None):
    """Analytical expressions for the monostatic target strength of a rectangular plate."""
    print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
    print("a >> lambda, ",a," >> ",wavelength)
    print("b >> lambda, ",b," >> ",wavelength)
    print("D = max(a,b)")
    D = max(a,b)
    print("range >> 2*D^2/lambda, ",target_range," >> ",2*D*D/wavelength)
    A = a*b
    TES = 20 * np.log10(A / (2*wavelength))
    print('Analytical on Beam TES = ', TES)

    if angles is None:
        return TES
    
    theta = np.radians(angles + 1e-6)
    beta = (2*np.pi/wavelength)*a*np.sin(theta)
    TES_angle = 10.*np.log10((((a*b)/(2*wavelength))**2)*((np.sin(beta)/beta)**2)*(0.5 + 0.5*np.cos(2*theta)))

    return TES_angle

def do_monostatic_rotation():
    """Do a monostatic rotation of the plate."""
    a = 3.0
    b = 2.0
    cp = 1480.0
    frequency = 10e3
    wavelength = cp / frequency
    target_range = 4000
    target = geo.make_rectangle(a,b)


    angles = np.linspace(0, 90, 91, endpoint=False)
    pnts = geo.generate_field_points(target_range, angles)
    res = mh.monostatic_iterated(pnts, target, cp, frequency)
    tes = mh.results_to_TES(res, target_range)
    tes_ana = mono_rectangle_tes(a,b,wavelength,target_range, angles)

    plt.figure()
    plt.plot(angles, tes, label="TES Comp (dB)")
    plt.plot(angles, tes_ana, label="TES Analytic (dB)")    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("TES at Field Points (dB)")
    plt.title("Mono static TES vs. Plate Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

    error = np.average(np.abs(tes - tes_ana))
    print("Average Error: ", error)
    return error

do_monostatic_rotation()
# api.render_openGL()


