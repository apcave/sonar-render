import ctypes
import numpy as np

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


def load_stl_mesh_to_cuda(stl_mesh, object_type, resolution=2e-3):
    """Loads a geometry that can be a source or a target to the CUDA library."""

    # Define the function signature in the shared library
    cpp_lib.load_geometry.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
    cpp_lib.load_geometry.restype = None


    num_facets = len(stl_mesh.vectors)
    flattened = stl_mesh.vectors.flatten()
    # Pass the array to C++
    v0_ptr = flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
 
    cpp_lib.load_geometry(v0_ptr, num_facets, object_type, resolution)

def render_cuda():
    """Pixelate the facets of the STL mesh."""

    # Define the function signature in the shared library
    cpp_lib.render_cuda.argtypes = []
    cpp_lib.render_cuda.restype = None

    cpp_lib.render_cuda()
    # Define the function signature in the shared library


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

def render_openGL(width, height, viewSettings, filename="output.png"):
    """Render the OpenGL window with the specified width and height."""
    cpp_lib.RenderOpenGL.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_float)]
    cpp_lib.RenderOpenGL.restype = None

    # Convert viewSettings to a ctypes float array
    viewSettings_arr = (ctypes.c_float * len(viewSettings))(*viewSettings)

    cpp_lib.RenderOpenGL(width, height, filename.encode('utf-8'),viewSettings_arr)

def TearDownCuda():
    """Tear down the CUDA model."""
    cpp_lib.TearDownCuda()

def project_source_points_to_objects():
    """Project source points to field points."""
    cpp_lib.ProjectSrcPointsToObjects()

def project_target_to_field_objects():
    """Project target points to field points."""
    cpp_lib.ProjectTargetToFieldObjects()

def project_target_to_field_points():
    """Project target points to field points."""
    cpp_lib.ProjectTargetToFieldObjects()

def project_target_to_target_objects():
    """Project target points to target objects."""
    cpp_lib.ProjectTargetToTargetObjects()