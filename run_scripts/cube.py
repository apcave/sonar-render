import TES_Model_API as api
import ModelHelpers as mh
import Geometry as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math








cp = 1480.0
frequency = 5e3
#frequency = 1e3
target_range = 000
angle_i = 0.0
target = geo.make_cube()
for i in range(4):
    target = geo.halve_facets(target)

#field_surface = geo.make_rectangle(10,10, False)
field_surface = geo.make_rectangle(9,14, False)

field_surface = geo.translate_stl_object(field_surface, [0, -0.1, 3])

for i in range(1):
    field_surface = geo.halve_facets(field_surface)


field_surface = geo.translate_stl_object(field_surface, [0, 0, -3])
target = geo.translate_stl_object(target, [0, 0, -3])

angle_i = [25]
# source_pnts= geo.generate_field_points(target_range, angle_i)
t = 20
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
