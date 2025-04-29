import TES_Model_API as api
import Geometry as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math







def bistatic_TES(a, b, wavelength, theta_i_deg, theta_s_deg):
    """
    Calculates bistatic TES for a rectangular plate.
    
    Parameters:
    a : float
        Plate length along x (meters)
    b : float
        Plate width along y (meters)
    wavelength : float
        Acoustic wavelength (meters)
    theta_i_deg : float or ndarray
        Incident angle in degrees (0 = broadside incidence)
    theta_s_deg : float or ndarray
        Scattering angle in degrees (0 = broadside scattering)
    
    Returns:
    TES_dB : float or ndarray
        Target Echo Strength in decibels
    """
    # Convert angles to radians
    theta_i = np.radians(theta_i_deg)
    theta_s = np.radians(theta_s_deg)

    # Wavenumber
    k = 2 * np.pi / wavelength

    # Define beta and gamma for diffraction terms
    beta = (np.pi * a / wavelength) * (np.sin(theta_s) - np.sin(theta_i))
    gamma = (np.pi * b / wavelength) * (np.cos(theta_s) - np.cos(theta_i))

    # Handle division by zero in sinc-like terms
    def sinc(x):
        return np.ones_like(x) if np.all(x == 0) else np.sinc(x / np.pi)

    sinc_beta = sinc(beta)
    sinc_gamma = sinc(gamma)

    # Compute TES
    TES = ((a * b) / (2 * wavelength))**2
    TES *= sinc_beta**2 * sinc_gamma**2
    TES *= ((np.cos(theta_i) + np.cos(theta_s)) / 2)**2

    TES_dB = 10 * np.log10(TES + 1e-12)  # avoid log(0)
    return TES_dB


a = 3.0
b = 2.0
cp = 1480.0
frequency = 5e3
target_range = 4000
angle_i = 20.0
target = geo.make_rectangle(a,b)
for i in range(6):
    target = geo.halve_facets(target)

api.load_stl_mesh_to_cuda(target, 0)

field_surface = geo.make_rectangle(10,10, False)
for i in range(4):
    field_surface = geo.halve_facets(field_surface)
api.load_stl_mesh_to_cuda(field_surface, 2)

#stl_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')
angle_i = [angle_i]
source_pnts= geo.generate_field_points(target_range, angle_i)

angles = np.linspace(-180, 180, 360, endpoint=False)
field_pnts= geo.generate_field_points(target_range, angles)


api.load_points_to_cuda(source_pnts, isSource=True)
api.load_points_to_cuda(field_pnts, isSource=False)
api.set_initial_conditions(cp, frequency, 0.0)

api.render_cuda()
field_vals = api.GetFieldPoints(len(field_pnts))

magnitudes = np.sqrt(field_vals[:, 0]**2 + field_vals[:, 1]**2)
mask = magnitudes > 1e-20
magnitudes = magnitudes[mask]
angles = angles[mask]
wavelength = cp / frequency
k = 2*np.pi/wavelength
A = a*b


db_values = 20 * np.log10(magnitudes)

print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print("a >> lambda, ",a," >> ",wavelength)
print("b >> lambda, ",b," >> ",wavelength)
print("D = max(a,b)")
D = max(a,b)
print("range >> 2*D^2/lambda, ",target_range," >> ",2*D*D/wavelength)

TES = 20 * np.log10(A / (2*wavelength))
# TES = 20*np.log10((k*A)/(4*np.pi))

atten = 20*np.log10(target_range)
print('Attenuation = ', atten)

index = np.where(angles == 0)[0][0] 
print("Modelled with Attenuation = ", db_values[index])
print('Analytical TES = ', TES)
print('Modelled TES = ', db_values[index] + 2* atten)

bistatic_TES = bistatic_TES(a, b, wavelength, angles, angle_i)


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

print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")

print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
api.render_openGL()
api.TearDownCuda()


