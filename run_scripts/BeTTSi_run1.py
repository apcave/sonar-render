import TES_Model_API as api
import ModelHelpers as mh
import Geometry as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import datetime










a = 3.0
b = 2.0
cp = 1480.0
frequency = 1e3
target_range = 4000
angle_i = 0.0
target = geo.load_stl_file('../hull_20cmMesh.stl')
#target = geo.rotate_stl_object(target, 'x', -20)
#target = geo.rotate_stl_object(target, 'y', 10)
#target = geo.rotate_stl_object(target, 'y', 10)


#geo.rotate_stl_object(target, 'z', 20)
#geo.rotate_stl_object(target, 'x', 45)
#geo.rotate_stl_object(target, 'y', 30)
# geo.save_mesh_to_stl(target, "trihedral_reflector.stl")
preview = False
a = 70*4
Area = a * a
length = 100*2
width = Area / length
field_surface = geo.make_rectangle(140*2,140*2, False)
geo.rotate_stl_object(field_surface, 'x', 90)
geo.rotate_stl_object(field_surface, 'z', 45)

for i in range(3):
    field_surface = geo.halve_facets(field_surface)

#145.39013504981995

angle_i = [0]
t = 1000
#source_pnts=[[0*4*t,0*3*t,9*t],[0,3*t,-9*t]]
source_pnts=[[0.2*t,1*t,0.1*t]]
angles = np.linspace(-180, 180, 361, endpoint=False)
field_pnts= geo.generate_field_points(target_range, angles)

projectTime = 0.1521
feildRes = 0.2
hoursComputation = 4
numPnts = (hoursComputation*3600 / projectTime)
length = np.sqrt(numPnts)*feildRes
print("Number of Field Points: ", numPnts)
print("Length of Field Surface: ", length)


api.load_points_to_cuda(source_pnts, isSource=True)
api.load_points_to_cuda(field_pnts, isSource=False)
api.set_initial_conditions(cp, frequency, 0.0)
api.load_stl_mesh_to_cuda(target, 0, 0.1) # 0 is for target object.
api.load_stl_mesh_to_cuda(field_surface, 2, feildRes) # 1 is for field surface.

mh.run_rendering(0,True)

field_vals = api.GetFieldPoints(len(field_pnts))

wavelength = cp / frequency
k = 2*np.pi/wavelength

a = 1

modeled_TES = mh.results_to_TES(field_vals, target_range)

print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print("a >> lambda, ",a," >> ",wavelength)
print("b >> lambda, ",b," >> ",wavelength)
print("D = max(a,b)")
D = max(a,b)
print("range >> 2*D^2/lambda, ",target_range," >> ",2*D*D/wavelength)


TES = 20 * np.log10((np.pi*a**2) / (wavelength**2))
# TES = 20*np.log10((k*A)/(4*np.pi))
print('Analytical TES = ', TES)
TES = 20 * np.log10((k*a**3)/2)
print('Analytical TES = ', TES)
print('ka << 1 ,', k, " << 1")




# index = np.where(angles == 0)[0][0] 
# print("Modelled with Attenuation = ", db_values[index])
# print('Analytical TES = ', TES)
# print('Modelled TES = ', db_values[index] + 2* atten)


# db_values  + 40*np.log10(Radius)    
# plt.figure()
# plt.plot(angles, modeled_TES, label="Field Values (dB)")
# #plt.plot(angles, bistatic_TES, label="Analytic (dB)")
# plt.xlabel("Angle (degrees)")
# plt.ylabel("Field Value (dB)")
# plt.title("Field Values vs. Angle")
# plt.grid(True)
# plt.legend()
# plt.show()

# Camera Postion
radius = 170.0
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
viewSettings[7] = 0.0
viewSettings[8] = 1.0

if preview:
    window_width = 800
    window_height = 600
else:
    window_width = 800 * 16
    window_height = 600 * 16
api.render_openGL(window_width, window_height, viewSettings, "BeTTSi.png")
#api.TearDownCuda()
