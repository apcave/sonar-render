
# Load the shared library
import ctypes
import numpy as np
import math
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

cpp_lib = ctypes.CDLL('./src/host/cuda_project.so')

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
    v0_ptr = points.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    print("Loading points to CUDA")

    cpp_lib.load_geometry.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    if isSource:
        cpp_lib.load_source_points(v0_ptr, num_points)
    else:
        cpp_lib.load_field_points(v0_ptr, num_points)


def load_stl_mesh_to_cuda(stl_mesh, isSource=False):
    """Loads a geometry that can be a source or a target to the CUDA library."""

    # Define the function signature in the shared library
    cpp_lib.load_geometry.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    cpp_lib.load_geometry.restype = None


    num_vertices = len(stl_mesh.v0)
    # Pass the array to C++
    v0_ptr = stl_mesh.v0.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v1_ptr = stl_mesh.v1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v2_ptr = stl_mesh.v2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cpp_lib.load_geometry(v0_ptr, v1_ptr, v2_ptr, num_vertices)

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
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [1.0, 1.0, 0.0],  # Vertex 2
        [0.0, 1.0, 0.0],  # Vertex 3
    ])

    vertices[:, 0] *= length
    vertices[:, 0] -= length/2  # Scale x-coordinates by length
    vertices[:, 1] *= width  # Scale y-coordinates by width
    vertices[:, 1] -= width/2

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

    # Save the mesh to an STL file
    plate.save('rectangular_plate.stl')

    print("STL file 'rectangular_plate.stl' created successfully!")
    return plate;

stl_mesh = make_rectangle(2.0,1.0)
#stl_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')
source_pnts = []
source_pnts.append([0.0,0.0,15.0])
angles = np.linspace(-180, 180, 360, endpoint=False)
field_pnts= generate_field_points(15, angles)

load_points_to_cuda(source_pnts, isSource=True)
load_points_to_cuda(field_pnts, isSource=False)
set_initial_conditions(1480.0, 1.0e3, 0.0)
load_stl_mesh_to_cuda(stl_mesh)
pixelate_facets()

plot_geometry(stl_mesh, source_pnts, field_pnts )

field_vals = GetFieldPoints(len(field_pnts))

magnitudes = np.sqrt(field_vals[:, 0]**2 + field_vals[:, 1]**2)
db_values = 20 * np.log10(magnitudes)

# Plot the data
plt.figure()
plt.plot(angles, db_values, label="Field Values (dB)")
plt.xlabel("Angle (degrees)")
plt.ylabel("Field Value (dB)")
plt.title("Field Values vs. Angle")
plt.grid(True)
plt.legend()
plt.show()


