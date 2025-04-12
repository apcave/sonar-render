from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Load the STL file
your_mesh = mesh.Mesh.from_file('./testing/rectangular_plate.stl')

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add the mesh to the plot
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
ax.auto_scale_xyz(scale, scale, scale)

# Show the plot
plt.show()