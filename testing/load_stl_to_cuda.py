
# Load the shared library
import ctypes
import numpy as np
import math
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

cpp_lib = ctypes.CDLL('./build/libcuda_project.so')

def set_initial_conditions(cp, frequency, attenuation):
    """ Sets the initial conditions of the CUDA model."""


    # Define the function signature in the shared library
    cpp_lib.set_initial_conditions.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    cpp_lib.set_initial_conditions.restype = None

    cpp_lib.set_initial_conditions(cp, frequency, attenuation)

def load_points_to_cuda(points, isSource=False):
    """ Loads points that can be a source or field points to the CUDA library."""
    points = np.array(points, dtype=np.float32)  # Ensure points is a NumPy array
    num_points = len(points)

    flattened = points.flatten()
    v0_ptr = flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cpp_lib.load_geometry.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    if isSource:
        cpp_lib.load_source_points(v0_ptr, num_points)
    else:
        cpp_lib.load_field_points(v0_ptr, num_points)


def load_stl_mesh_to_cuda(stl_mesh, isSource=False):
    """Loads a geometry that can be a source or a target to the CUDA library."""

    # Define the function signature in the shared library
    cpp_lib.load_geometry.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    cpp_lib.load_geometry.restype = None


    num_facets = len(stl_mesh.vectors)
    flattened = stl_mesh.vectors.flatten()
    # Pass the array to C++
    v0_ptr = flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
 
    cpp_lib.load_geometry(v0_ptr, num_facets)

def pixelate_facets():
    """Pixelate the facets of the STL mesh."""

    # Define the function signature in the shared library
    cpp_lib.pixelate_facets.argtypes = []
    cpp_lib.pixelate_facets.restype = None

    cpp_lib.pixelate_facets()
    # Define the function signature in the shared library

def generate_field_points(radius, angles):
    """Generate field points in the x-z plane at 1-degree spacing."""
    field_points = []
    for i in angles:
        angle = math.radians(i+90)  # Convert degrees to radians
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = 0.0  # y-coordinate is 0 in the x-z plane
        field_points.append([x, y, z])
    return field_points

def plot_geometry(stl_mesh, source_pnts, field_pnts ):
    """Plot the geometry using matplotlib."""
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the mesh to the plot
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))

    # Plot the source point as a large red dot
    source_pnts = np.array(source_pnts)
    ax.scatter(source_pnts[:, 0], source_pnts[:, 1], source_pnts[:, 2], color='red', s=100, label='Source Point')

    # Plot the field points as green dots
    field_pnts = np.array(field_pnts)
    ax.scatter(field_pnts[:, 0], field_pnts[:, 1], field_pnts[:, 2], color='green', s=10, label='Field Points')


    # Auto scale to the mesh size
    scale = stl_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Manually set axis limits to ensure all points are visible
    ax.set_xlim([-20, 20])  # Adjust based on the range of your field points and mesh
    ax.set_ylim([-20, 20])  # Adjust as needed (y is 0 for field points in the x-z plane)
    ax.set_zlim([-5, 20])   # Adjust to include the source point and field points


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()

def GetFieldPoints(NumFieldPnts):
    """Get the field points from the CUDA library."""
    # Define the function signature in the shared library
    cpp_lib.GetFieldPointPressures.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    cpp_lib.GetFieldPointPressures.restype = None

    # Create an array to hold the field points
    field_points = np.zeros((NumFieldPnts, 2), dtype=np.double)
    field_points_ptr = field_points.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    cpp_lib.GetFieldPointPressures(field_points_ptr, NumFieldPnts)

    return field_points

def make_rectangle(length, width):
    # Define the vertices of the rectangular plate
    vertices = np.array([
        [-length/2, -width/2, 0.0],  # Vertex 0
        [ length/2, -width/2, 0.0],  # Vertex 1
        [ length/2, width/2, 0.0],  # Vertex 2
        [-length/2, width/2, 0.0],  # Vertex 3
    ])

    # Define the two triangular facets using the vertices
    # Each row represents a triangle (3 vertices)
    facets = np.array([
        [vertices[0], vertices[1], vertices[2]],  # First triangle
        [vertices[0], vertices[2], vertices[3]],  # Second triangle
    ])

    # Create the mesh
    plate = mesh.Mesh(np.zeros(facets.shape[0], dtype=mesh.Mesh.dtype))
    for i, facet in enumerate(facets):
        plate.vectors[i] = facet

    #print("Vertices:")
    #print(plate.vectors)
    
    # Save the mesh to an STL file
    plate.save('rectangular_plate.stl')

    print("STL file 'rectangular_plate.stl' created successfully!")
    return plate

def render_openGL():
    """Render the OpenGL window."""

    cpp_lib.RenderOpenGL()


a = 3.0
b = 2.0
cp = 1480.0
frequency = 2.0e3
Radius = 50.0

stl_mesh = make_rectangle(a,b)
#stl_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')
source_pnts = []
source_pnts.append([0.0,0.0,Radius])
angles = np.linspace(-180, 180, 360, endpoint=False)
field_pnts= generate_field_points(Radius, angles)

load_points_to_cuda(source_pnts, isSource=True)
load_points_to_cuda(field_pnts, isSource=False)
set_initial_conditions(cp, frequency, 0.0)
load_stl_mesh_to_cuda(stl_mesh)
#render_openGL()
pixelate_facets()

#plot_geometry(stl_mesh, source_pnts, field_pnts )

field_vals = GetFieldPoints(len(field_pnts))



magnitudes = np.sqrt(field_vals[:, 0]**2 + field_vals[:, 1]**2)
magnitudes = np.sqrt(magnitudes)
magnitudes = np.where(magnitudes == 0, np.finfo(float).eps, magnitudes)


db_values = 20 * np.log10(magnitudes)

wavelength = cp / frequency
A = a*b
TES = 10*np.log10((4*np.pi*A**2)/(wavelength**2))
print('Target Strength = ', TES)
EchoLevel = TES - 2 * 20 * np.log10(Radius)
print('Echo Level = ', EchoLevel)


wavelength = cp / frequency
A = a*b
TES = 10*np.log10((4*np.pi*A**2)/(wavelength**2))
TL = 2 * 20 * np.log10(Radius)
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print('Target Echo Strength = ', TES)
print('Transmission Loss = ', TL)
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")

# Using the Cross-Sectional Area
CrossSection = ((4*np.pi)*(A**2))/(wavelength**2)
print('Cross Section = ', CrossSection)
PressureRatio = (CrossSection/ (4*np.pi*Radius**2))**0.5
print('Pressure Ratio = ', PressureRatio)
PressureRatio_db = 20*np.log10(PressureRatio)
print('Pressure Ratio (dB) = ', PressureRatio_db)
TransmissionLoss_db = 20*np.log10((1/(4*np.pi*Radius**2)))
print('Transmission Loss = ', TransmissionLoss_db)
EchoRatio_db = PressureRatio_db + TransmissionLoss_db
EchoRatio = PressureRatio/(4*np.pi*Radius**2)
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print( "This is close to the modelled value.")
print('Echo Ratio = ', 20*np.log10(EchoRatio))
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")

wavelength = cp / frequency
A = a*b
TES = 10*np.log10(4*np.pi*A**2/wavelength**2)
print('Target Strength = ', TES)
EchoLevel = TES - 2 * 20 * np.log10(Radius)
print('Echo Level = ', EchoLevel)

modelled_TES = -41.25 + 40*np.log10(Radius)
print('Modelled Target Strength = ', modelled_TES)



# Plot the data
if True:
    # db_values  + 40*np.log10(Radius)    
    plt.figure()
    plt.plot(angles, db_values, label="Field Values (dB)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Field Value (dB)")
    plt.title("Field Values vs. Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

# 

