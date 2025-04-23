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
    api.set_initial_conditions(cp, frequency, 0.0)
    api.load_stl_mesh_to_cuda(object)
    

    p_reflect = []
    for pnt in pnts:
        print("Projection Point: ", pnt)
        api.load_points_to_cuda(pnt, isSource=True)
        api.load_points_to_cuda(pnt, isSource=False)
        api.render_cuda()
        field_vals = api.GetFieldPoints(1)
        p_reflect.append(field_vals)
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
    if xvals is not None:
        xvals = xvals[mask]
    atten = 20*np.log10(Radius)
    db_values = 20 * np.log10(magnitudes) + atten
    
    return db_values, xvals

a = 3.0
b = 2.0
cp = 1480.0
frequency = 10e3
range = 4000

stl_mesh = geo.make_rectangle(a,b)
angles = np.linspace(-180, 180, 360, endpoint=False)
pnts = geo.generate_field_points(range, angles)
res = monostatic_iterated(pnts, object, cp, frequency)

tes, angles = results_to_TES(res, range, angles)

plt.figure()
plt.plot(angles, tes, label="TES (dB)")
plt.xlabel("Angle (degrees)")
plt.ylabel("TES at Field Points (dB)")
plt.title("Mono static TES vs. Plate Angle")
plt.grid(True)
plt.legend()
plt.show()

exit()

#stl_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')
source_pnts = []



source_pnts.append([0.0,0.0,Radius])
angles = np.linspace(-180, 180, 360, endpoint=False)
field_pnts= geo.generate_field_points(Radius, angles)

print("Loading CUDA")
api.load_points_to_cuda(source_pnts, isSource=True)
api.load_points_to_cuda(field_pnts, isSource=False)
api.set_initial_conditions(cp, frequency, 0.0)
api.load_stl_mesh_to_cuda(stl_mesh)
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
print("Radius >> 2*D^2/lambda, ",Radius," >> ",2*D*D/wavelength)

TES = 20 * np.log10(A / (2*wavelength))
# TES = 20*np.log10((k*A)/(4*np.pi))

atten = 20*np.log10(Radius)
print('Attenuation = ', atten)

index = np.where(angles == 0)[0][0] 
print("Modelled with Attenuation = ", db_values[index])
print('Analytical TES = ', TES)
print('Modelled TES = ', db_values[index] + 2* atten)



# Plot the data
if True:
    # db_values  + 40*np.log10(Radius)    
    plt.figure()
    plt.plot(angles, db_values + 2 * atten, label="Field Values (dB)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Field Value (dB)")
    plt.title("Field Values vs. Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

# 
#api.render_openGL()
api.TearDownCuda()