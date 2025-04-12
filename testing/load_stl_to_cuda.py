
# Load the shared library
import ctypes
import numpy as np
from stl import mesh
cpp_lib = ctypes.CDLL('./src/host/cuda_project.so')

def set_initial_conditions(cp, frequency, attenuation):
    """ Sets the initial conditions of the CUDA model."""


    # Define the function signature in the shared library
    cpp_lib.set_initial_conditions.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    cpp_lib.set_initial_conditions.restype = None

    cpp_lib.set_initial_conditions(cp, frequency, attenuation)

def load_points_to_cuda(points, isSource=False):
    """ Loads points that can be a source or field points to the CUDA library."""
    print("Loading points to CUDA")

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



stl_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')


set_initial_conditions(1480.0, 1.0e3, 0.0)
load_stl_mesh_to_cuda(stl_mesh)
pixelate_facets()
