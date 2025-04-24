import TES_Model_API as api
import Geometry as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math









a = 3.0
b = 2.0
cp = 1480.0
frequency = 50e3
range = 4000
angle_i = 0.0
target = geo.load_stl_file("./testing/sphere_1m_radius.stl")

angle_i = [180]
source_pnts= geo.generate_field_points(range, angle_i)
angle_i = [-angle_i[0]]
angles = np.linspace(-180, 180, 360, endpoint=False)
field_pnts= geo.generate_field_points(range, angles)


api.load_points_to_cuda(source_pnts, isSource=True)
api.load_points_to_cuda(field_pnts, isSource=False)
api.set_initial_conditions(cp, frequency, 0.0)
api.load_stl_mesh_to_cuda(target)
api.render_cuda()
field_vals = api.GetFieldPoints(len(field_pnts))

magnitudes = np.sqrt(field_vals[:, 0]**2 + field_vals[:, 1]**2)
mask = magnitudes > 1e-20
magnitudes = magnitudes[mask]
angles = angles[mask]
wavelength = cp / frequency
k = 2*np.pi/wavelength

a = 1

db_values = 20 * np.log10(magnitudes)

print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print("a >> lambda, ",a," >> ",wavelength)
print("b >> lambda, ",b," >> ",wavelength)
print("D = max(a,b)")
D = max(a,b)
print("range >> 2*D^2/lambda, ",range," >> ",2*D*D/wavelength)


TES = 20 * np.log10((np.pi*a**2) / (wavelength**2))
# TES = 20*np.log10((k*A)/(4*np.pi))
print('Analytical TES = ', TES)
TES = 20 * np.log10((k*a**3)/2)
print('Analytical TES = ', TES)
print('ka << 1 ,', k, " << 1")


atten = 20*np.log10(range)
print('Attenuation = ', atten)

index = np.where(angles == 0)[0][0] 
print("Modelled with Attenuation = ", db_values[index])
print('Analytical TES = ', TES)
print('Modelled TES = ', db_values[index] + 2* atten)


# db_values  + 40*np.log10(Radius)    
plt.figure()
plt.plot(angles, db_values + 2 * atten, label="Field Values (dB)")
#plt.plot(angles, bistatic_TES, label="Analytic (dB)")
plt.xlabel("Angle (degrees)")
plt.ylabel("Field Value (dB)")
plt.title("Field Values vs. Angle")
plt.grid(True)
plt.legend()
plt.show()


api.render_openGL()
#api.TearDownCuda()
