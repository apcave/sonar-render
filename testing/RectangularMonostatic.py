import TES_Model_API as api
import Geometry as geo

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
        print("Projection Point: ", pnt)
        api.set_initial_conditions(cp, frequency, 0.0)
        api.load_stl_mesh_to_cuda(object)        
        api.load_points_to_cuda([pnt], isSource=True)
        api.load_points_to_cuda([pnt], isSource=False)
        api.render_cuda()
        field_vals = api.GetFieldPoints(1)
        p_reflect.append([float(field_vals[0][0]), float(field_vals[0][1])])
        api.TearDownCuda()

    return p_reflect

def results_to_TES(res, range, xvals = None):
    """Convert the results to Target Strength."""
    # Convert the results to numpy array
    res = np.array(res)
    # Calculate the magnitudes
    magnitudes = np.sqrt(res[:, 0]**2 + res[:, 1]**2)
    mask = magnitudes > 1e-20
    magnitudes = magnitudes[mask]
    atten = 20*np.log10(range)
    db_values = 20 * np.log10(magnitudes) + 2 * atten

    if xvals is not None:
        xvals = xvals[mask]

    
    return db_values, xvals


a = 3.0
b = 2.0
cp = 1480.0
frequency = 10e3
wavelength = cp / frequency
range = 4000
target = geo.make_rectangle(a,b)

def get_analytical_for_rectangle(angles = None):
    print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
    print("a >> lambda, ",a," >> ",wavelength)
    print("b >> lambda, ",b," >> ",wavelength)
    print("D = max(a,b)")
    D = max(a,b)
    print("range >> 2*D^2/lambda, ",range," >> ",2*D*D/wavelength)
    A = a*b
    TES = 20 * np.log10(A / (2*wavelength))
    print('Analytical on Beam TES = ', TES)

    if angles is None:
        return
    
    theta = np.radians(angles)
    beta = (2*np.pi/wavelength)*a*np.sin(theta)
    TES_angle = 10.*np.log10((((a*b)/(2*wavelength))**2)*((np.sin(beta)/beta)**2)*(0.5 + 0.5*np.cos(2*theta)))


    # TES_angle = 20 * np.log10(cos_inc * (A / (2*wavelength)))
    return TES_angle

def do_monostatic_rotation():
    """Do a monostatic rotation of the plate."""
    
    angles = np.linspace(0, 180, 181, endpoint=False)
    pnts = geo.generate_field_points(range, angles)
    res = monostatic_iterated(pnts, target, cp, frequency)

    tes, angles = results_to_TES(res, range, angles)
    tes_ana = get_analytical_for_rectangle(angles)

    plt.figure()
    plt.plot(angles, tes, label="TES Comp (dB)")
    plt.plot(angles, tes_ana, label="TES Analytic (dB)")    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("TES at Field Points (dB)")
    plt.title("Mono static TES vs. Plate Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

do_monostatic_rotation()


# def bistatic_TES(a, b, wavelength, theta_i_deg, theta_s_deg):
#     """
#     Calculates bistatic TES for a rectangular plate.
    
#     Parameters:
#     a : float
#         Plate length along x (meters)
#     b : float
#         Plate width along y (meters)
#     wavelength : float
#         Acoustic wavelength (meters)
#     theta_i_deg : float or ndarray
#         Incident angle in degrees (0 = broadside incidence)
#     theta_s_deg : float or ndarray
#         Scattering angle in degrees (0 = broadside scattering)
    
#     Returns:
#     TES_dB : float or ndarray
#         Target Echo Strength in decibels
#     """
#     # Convert angles to radians
#     theta_i = np.radians(theta_i_deg)
#     theta_s = np.radians(theta_s_deg)

#     # Wavenumber
#     k = 2 * np.pi / wavelength

#     # Define beta and gamma for diffraction terms
#     beta = (np.pi * a / wavelength) * (np.sin(theta_s) - np.sin(theta_i))
#     gamma = (np.pi * b / wavelength) * (np.cos(theta_s) - np.cos(theta_i))

#     # Handle division by zero in sinc-like terms
#     def sinc(x):
#         return np.ones_like(x) if np.all(x == 0) else np.sinc(x / np.pi)

#     sinc_beta = sinc(beta)
#     sinc_gamma = sinc(gamma)

#     # Compute TES
#     TES = ((a * b) / (2 * wavelength))**2
#     TES *= sinc_beta**2 * sinc_gamma**2
#     TES *= ((np.cos(theta_i) + np.cos(theta_s)) / 2)**2

#     TES_dB = 10 * np.log10(TES + 1e-12)  # avoid log(0)
#     return TES_dB

def bistatic_tes_rectangular_plate(a, b, wavelength, theta_i_deg, theta_s_deg):
    """
    Calculate bistatic TES of a rectangular plate using Kirchhoff approximation.

    Parameters:
    - a, b: dimensions of the plate (meters)
    - wavelength: wavelength (meters)
    - theta_i_deg: incident angle from normal (degrees)
    - theta_s_deg: scattering angle from normal (degrees)

    Returns:
    - TES in dB
    """
    k = 2 * np.pi / wavelength
    theta_i = np.radians(theta_i_deg)
    theta_s = np.radians(theta_s_deg)

    X = (k * a / 2) * (np.sin(theta_s) - np.sin(theta_i))
    Y = (k * b / 2) * (np.cos(theta_s) - np.cos(theta_i))

    sinc_X = np.sinc(X / np.pi)  # numpy sinc is sin(pi*x)/(pi*x)
    sinc_Y = np.sinc(Y / np.pi)

    cos_mid_angle = np.cos((theta_i + theta_s) / 2)
    amplitude = (a * b / wavelength) * cos_mid_angle * sinc_X * sinc_Y

    tes_linear = amplitude**2
    tes_db = 10 * np.log10(tes_linear + 1e-12)  # +1e-12 to avoid log(0)
    return tes_db


#stl_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')
source_pnts = []


angle_i = [30.0]
source_pnts= geo.generate_field_points(range, angle_i)
angle_i = [-30.0]
angles = np.linspace(-180, 180, 360, endpoint=False)
field_pnts= geo.generate_field_points(range, angles)

print("Loading CUDA")
api.load_points_to_cuda(source_pnts, isSource=True)
api.load_points_to_cuda(field_pnts, isSource=False)
api.set_initial_conditions(cp, frequency, 0.0)
api.load_stl_mesh_to_cuda(target)
#render_openGL()
print("Doing CUDA Calculation")
api.render_cuda()
print("Clearing the GPU...")


print("<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# plot_geometry(stl_mesh, source_pnts, field_pnts )

field_vals = api.GetFieldPoints(len(field_pnts))

print("Tear Down Cuda")
print("Debug Varible :  ", field_vals[0,0])

magnitudes = np.sqrt(field_vals[:, 0]**2 + field_vals[:, 1]**2)
mask = magnitudes > 1e-20
magnitudes = magnitudes[mask]
angles = angles[mask]
wavelength = cp / frequency
k = 2*np.pi/wavelength
A = a*b

print("Test 1 : ",(k*A)/(4*np.pi))
print("Test 2 : ",field_vals[0,0])

db_values = 20 * np.log10(magnitudes)


print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print("a >> lambda, ",a," >> ",wavelength)
print("b >> lambda, ",b," >> ",wavelength)
print("D = max(a,b)")
D = max(a,b)
print("range >> 2*D^2/lambda, ",range," >> ",2*D*D/wavelength)

TES = 20 * np.log10(A / (2*wavelength))
# TES = 20*np.log10((k*A)/(4*np.pi))

atten = 20*np.log10(range)
print('Attenuation = ', atten)

index = np.where(angles == 0)[0][0] 
print("Modelled with Attenuation = ", db_values[index])
print('Analytical TES = ', TES)
print('Modelled TES = ', db_values[index] + 2* atten)

bistatic_TES = bistatic_tes_rectangular_plate(a, b, wavelength, angle_i, angles )

# Plot the data
if True:
    # db_values  + 40*np.log10(Radius)    
    plt.figure()
    plt.plot(angles, db_values + 2 * atten, label="Field Values (dB)")
    plt.plot(angles, bistatic_TES, label="Analytic (dB)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Field Value (dB)")
    plt.title("Field Values vs. Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

# 
#api.render_openGL()
api.TearDownCuda()

