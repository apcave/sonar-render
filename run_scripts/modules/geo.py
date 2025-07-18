
import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import trimesh
from trimesh.creation import box

def save_mesh_to_stl(stl_mesh, file_path):
    """
    Saves an STL mesh to a file.

    Parameters:
    stl_mesh : mesh.Mesh
        The STL mesh object to save.
    file_path : str
        The path to save the STL file.
    """
    try:
        stl_mesh.save(file_path)
        print(f"Mesh successfully saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving mesh to '{file_path}': {e}")

def plot_geometry(stl_mesh, source_pnts, field_pnts ):
    """Plot the geometry using matplotlib."""
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the mesh to the plot
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))

    # Plot the source point as a large red dot
    if len(source_pnts) > 0:
        source_pnts = np.array(source_pnts)
        ax.scatter(source_pnts[:, 0], source_pnts[:, 1], source_pnts[:, 2], color='red', s=100, label='Source Point')

    # Plot the field points as green dots
    if len(field_pnts) > 0:
        field_pnts = np.array(field_pnts)
        ax.scatter(field_pnts[:, 0], field_pnts[:, 1], field_pnts[:, 2], color='green', s=10, label='Field Points')


    # Auto scale to the mesh size
    scale = stl_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Manually set axis limits to ensure all points are visible
    ax.set_xlim([-2, 2])  # Adjust based on the range of your field points and mesh
    ax.set_ylim([-2, 2])  # Adjust as needed (y is 0 for field points in the x-z plane)
    ax.set_zlim([-2, 2])   # Adjust to include the source point and field points


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()

def generate_field_points(radius, angles):
    """Generate field points in the x-z plane at 1-degree spacing."""
    field_points = []
    for i in angles:
        angle = math.radians(i)  # Convert degrees to radians
        x = radius * math.sin(angle)
        z = radius * math.cos(angle)
        y = 0.0  # y-coordinate is 0 in the x-z plane
        field_points.append([x, y, z])
    return field_points

def make_rectangle(length, width, xy_plane=True):
    # Define the vertices of the rectangular plate


    if xy_plane:
        vertices = np.array([
            [-length/2, -width/2, 0.0],  # Vertex 0
            [ length/2, -width/2, 0.0],  # Vertex 1
            [ length/2, width/2, 0.0],  # Vertex 2
            [-length/2, width/2, 0.0],  # Vertex 3
        ])
    else:
        # Alternatively, define the vertices in the x-z plane
        vertices = np.array([
            [-length/2, 0.0, -width/2],  # Vertex 0
            [ length/2, 0.0, -width/2],  # Vertex 1
            [ length/2, 0.0, width/2],   # Vertex 2
            [-length/2, 0.0, width/2],   # Vertex 3
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
    # plate.save('rectangular_plate.stl')

    return plate

def make_right_angle_triangle(x_length, y_length):
    """
    Creates a right-angle triangle mesh in the x-y plane with the right angle at the origin.
    The hypotenuse runs from (x_length, 0, 0) to (0, y_length, 0).

    Parameters:
    x_length : float
        Length along the x-axis.
    y_length : float
        Length along the y-axis.

    Returns:
    mesh.Mesh
        The STL mesh of the triangle.
    """
    vertices = np.array([
        [0, 0, 0],                # Right angle at origin
        [x_length, 0, 0],         # Along x-axis
        [0, y_length, 0],         # Along y-axis
    ])

    # One triangle: vertices 0, 1, 2
    facets = np.array([
        [vertices[0], vertices[1], vertices[2]],
    ])

    triangle = mesh.Mesh(np.zeros(facets.shape[0], dtype=mesh.Mesh.dtype))
    for i, facet in enumerate(facets):
        triangle.vectors[i] = facet

    return triangle

def load_stl_file(file_path):
    """Load an STL file and return the mesh object."""
    try:
        stl_mesh = mesh.Mesh.from_file(file_path)
        print(f"STL file '{file_path}' loaded successfully.")
        return stl_mesh
    except Exception as e:
        print(f"Error loading STL file '{file_path}': {e}")
        return None
    

def halve_facets(stl_mesh):
    """Halve all facets in an STL mesh while maintaining normals."""
    # Create a list to store the new facets
    new_vectors = []
    new_normals = []

    for i in range(len(stl_mesh.vectors)):
        # Get the vertices of the current triangle
        v1, v2, v3 = stl_mesh.vectors[i]

        # Calculate the midpoints of each edge
        mid12 = (v1 + v2) / 2
        mid23 = (v2 + v3) / 2
        mid31 = (v3 + v1) / 2

        # Create two new triangles using the midpoints
        # Triangle 1: v1, mid12, mid31
        new_vectors.append([v1, mid12, mid31])
        # Triangle 2: mid12, v2, mid23
        new_vectors.append([mid12, v2, mid23])
        # Triangle 3: mid23, v3, mid31
        new_vectors.append([mid23, v3, mid31])
        # Triangle 4: mid12, mid23, mid31
        new_vectors.append([mid12, mid23, mid31])

        # Use the original normal for all new triangles
        normal = stl_mesh.normals[i]
        new_normals.extend([normal, normal, normal, normal])

    # Convert the new vectors and normals to numpy arrays
    new_vectors = np.array(new_vectors)
    new_normals = np.array(new_normals)

    # Create a new mesh with the new facets
    new_mesh = mesh.Mesh(np.zeros(new_vectors.shape[0], dtype=mesh.Mesh.dtype))
    new_mesh.vectors = new_vectors
    new_mesh.normals = new_normals

    return new_mesh    

def make_cube():
        # Define the vertices of the cube (1 meter cube)
    # Define the 8 vertices of the cube
    vertices = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ])

    # Each face is made of two triangles, vertices in CCW order as seen from outside
    faces = np.array([
        # Bottom (z=0)
        [2, 1, 0],
        [3, 2, 0],
    
        # # Top (z=1)
        [5, 6, 4],
        [6, 7, 4],
        # # Front (y=0)
        [0, 4, 5],
        [0, 5, 1],
        # # Back (y=1)
        [6, 2, 3],
        [7, 6, 3],
        # # Left (x=0)
        [7, 3, 0],
        [4, 7, 0],
        # Right (x=1)
        [6, 5, 1],
        [2, 6, 1],
    ])


    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j], :]

    return cube

def join_meshes(mesh1, mesh2):
    """
    Joins two STL meshes into a single mesh.

    Parameters:
    mesh1 : mesh.Mesh
        The first STL mesh.
    mesh2 : mesh.Mesh
        The second STL mesh.

    Returns:
    combined_mesh : mesh.Mesh
        The combined STL mesh.
    """
    # Concatenate the data of both meshes
    combined_data = np.concatenate([mesh1.data, mesh2.data])

    # Create a new mesh with the combined data
    combined_mesh = mesh.Mesh(combined_data)

    return combined_mesh

def clone_mesh(original_mesh):
    """
    Creates a clone of an STL mesh.

    Parameters:
    original_mesh : mesh.Mesh
        The original STL mesh to clone.

    Returns:
    cloned_mesh : mesh.Mesh
        A new STL mesh that is a clone of the original.
    """
    # Create a new mesh with the same data as the original
    cloned_mesh = mesh.Mesh(np.copy(original_mesh.data))
    return cloned_mesh

def create_plate(length, width, thickness=0.01):
    vertices = np.array([
        [-length / 2, -width / 2, 0],  # Bottom-left corner
        [ length / 2, -width / 2, 0],  # Bottom-right corner
        [ length / 2,  width / 2, 0],  # Top-right corner
        [-length / 2,  width / 2, 0],  # Top-left corner
        [-length / 2, -width / 2, thickness],  # Bottom-left corner (top face)
        [ length / 2, -width / 2, thickness],  # Bottom-right corner (top face)
        [ length / 2,  width / 2, thickness],  # Top-right corner (top face)
        [-length / 2,  width / 2, thickness],  # Top-left corner (top face)
    ])

    # Define the faces of the plate
    faces = np.array([
       [2, 1, 0], [3, 2, 0],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Front face
        [0, 1, 5], [0, 5, 4],  # Side faces
       [1, 2, 6], [1, 6, 5],   # RHS
        [2, 3, 7], [2, 7, 6],   # TOP
        [3, 0, 4], [3, 4, 7],
    ])

    # Create the mesh
    plate = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            plate.vectors[i][j] = vertices[face[j], :]

    return plate


def create_plate_triangle(length, width, thickness=0.01):
    """
    Creates a triangular plate (prism) mesh with the given base length, width, and thickness.
    The triangle is in the x-y plane, extruded along the z-axis.

    Parameters:
        length: base length of the triangle (x-axis)
        width: height of the triangle (y-axis)
        thickness: extrusion along z-axis

    Returns:
        plate: mesh.Mesh object
    """
    import numpy as np
    from stl import mesh

    # Base triangle (z=0)
    v0 = [-length / 2, -width / 2, 0]
    v1 = [ length / 2, -width / 2, 0]
    v2 = [0, 0, 0]
    # Top triangle (z=thickness)
    v3 = [-length / 2, -width / 2, thickness]
    v4 = [ length / 2, -width / 2, thickness]
    v5 = [0, 0, thickness]

    vertices = np.array([v0, v1, v2, v3, v4, v5])

    # 8 faces: 2 for each triangle, 3 sides (2 triangles each)
    faces = np.array([
        [2, 1, 0],  # Bottom triangle
        [4, 5, 3],  # Top triangle (note winding for outward normals)
        [0, 1, 4], [0, 4, 3],  # Side 1
        [1, 2, 5], [1, 5, 4],  # Side 2
        [2, 0, 3], [2, 3, 5],  # Side 3
    ])

    # Create the mesh
    plate = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            plate.vectors[i][j] = vertices[face[j], :]

    return plate



def create_reflector(width, thickness):
    
    side_length = (2*((width/2)**2))**0.5
    plate_1 = create_plate_triangle(side_length, width, thickness)
    plate_1 = rotate_stl_object(plate_1, 'x', 45)

    plate_2 = create_plate_triangle(side_length, width, thickness)
    plate_2 = rotate_stl_object(plate_2, 'x', -45)


    
    plate_3 = create_plate_triangle(side_length, width, thickness)
    plate_3 = rotate_stl_object(plate_3, 'y', 90)
    plate_3 = rotate_stl_object(plate_3, 'z', 45)

    plate_4 = create_plate_triangle(side_length, width, thickness)
    plate_4 = rotate_stl_object(plate_4, 'y', 90)
    plate_4 = rotate_stl_object(plate_4, 'z', -45)

    reflector = join_meshes(plate_1, plate_2)
    reflector = join_meshes(reflector, plate_3)
    reflector = join_meshes(reflector, plate_4)  

    reflector = rotate_stl_object(reflector, 'x', -90)
 
    return reflector


def create_trihedral_reflector(side_length, thickness):
        
    plate_xy = create_plate(side_length, side_length, thickness)
    plate_xy = rotate_stl_object(plate_xy, 'z', 45)
    plate_xz = create_plate(side_length, side_length, thickness)
    plate_xz = rotate_stl_object(plate_xz, 'z', 45)
    plate_xz = rotate_stl_object(plate_xz, 'y', 90)
    plate_yz = create_plate(side_length, side_length, thickness)
    plate_yz = rotate_stl_object(plate_yz, 'z', 45)
    plate_yz = rotate_stl_object(plate_yz, 'x', 90)


    reflector = join_meshes(plate_xy, plate_xz)
    reflector = join_meshes(reflector, plate_yz)

    reflector = rotate_stl_object(reflector, 'z', 45)
    reflector = rotate_stl_object(reflector, 'x', 45)    

    #plate_xz = create_plate(side_length, thickness, 'xz')
    #plate_yz = create_plate(side_length, thickness, 'yz')

    #plate_xz.translate([0, -side_length / 2, 0])  # Move xz plate to align with xy
    #plate_yz.translate([-side_length / 2, 0, 0])  # Move yz plate to align with xy    

    
    return reflector

def rotate_stl_object(stl_mesh, axis, angle_degrees):
    """
    Rotates an STL object about the origin.

    Parameters:
    stl_mesh : mesh.Mesh
        The STL mesh object to rotate.
    axis : str
        The axis to rotate around ('x', 'y', or 'z').
    angle_degrees : float
        The angle to rotate in degrees.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Define rotation matrices for each axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Apply the rotation matrix to all vectors in the STL mesh
    stl_mesh.vectors = np.dot(stl_mesh.vectors, rotation_matrix.T)

    return stl_mesh

def translate_stl_object(stl_mesh, translation_vector):
    """
    Translates an STL mesh by a given 3D vector.

    Parameters:
    stl_mesh : mesh.Mesh
        The STL mesh object to translate.
    translation_vector : array-like or list or np.ndarray
        The [dx, dy, dz] translation vector.

    Returns:
    mesh.Mesh
        The translated STL mesh (in-place).
    """
    translation_vector = np.asarray(translation_vector)
    stl_mesh.vectors += translation_vector
    return stl_mesh


