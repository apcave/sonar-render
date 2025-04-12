from stl import mesh
import numpy as np

# Define the vertices of the rectangular plate
vertices = np.array([
    [0, 0, 0],  # Vertex 0
    [1, 0, 0],  # Vertex 1
    [1, 1, 0],  # Vertex 2
    [0, 1, 0],  # Vertex 3
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

# Save the mesh to an STL file
plate.save('rectangular_plate.stl')

print("STL file 'rectangular_plate.stl' created successfully!")