
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
        angle = math.radians(i+90)  # Convert degrees to radians
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = 0.0  # y-coordinate is 0 in the x-z plane
        field_points.append([x, y, z])
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
    # plate.save('rectangular_plate.stl')

    return plate