import TES_Model_API as api
import ModelHelpers as mh
import Geometry as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math









a = 3.0
b = 2.0
cp = 1480.0
frequency = 2e3
target_range = 4000
angle_i = 0.0
target = geo.create_trihedral_reflector(1, 0.01)
target = geo.rotate_stl_object(target, 'x', -20)
target = geo.rotate_stl_object(target, 'y', 10)
for i in range(4):
    target = geo.halve_facets(target)

#geo.rotate_stl_object(target, 'z', 20)
#geo.rotate_stl_object(target, 'x', 45)
#geo.rotate_stl_object(target, 'y', 30)
# geo.save_mesh_to_stl(target, "trihedral_reflector.stl")


field_surface = geo.make_rectangle(5,5, False)
field_surface = geo.translate_stl_object(field_surface, [0, -0.6, 1])

for i in range(3):
    field_surface = geo.halve_facets(field_surface)


angle_i = [0]
t = 20
#source_pnts=[[0*4*t,0*3*t,9*t],[0,3*t,-9*t]]
source_pnts=[[4*t,3*t,9*t]]
angles = np.linspace(-180, 180, 361, endpoint=False)
field_pnts= geo.generate_field_points(target_range, angles)


api.load_points_to_cuda(source_pnts, isSource=True)
api.load_points_to_cuda(field_pnts, isSource=False)
api.set_initial_conditions(cp, frequency, 0.0)
api.load_stl_mesh_to_cuda(target, 0) # 0 is for target object.
api.load_stl_mesh_to_cuda(field_surface, 2) # 1 is for field surface.

api.render_cuda()
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


api.render_openGL()
#api.TearDownCuda()
