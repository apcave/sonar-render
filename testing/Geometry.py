
import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

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