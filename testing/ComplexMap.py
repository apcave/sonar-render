import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Generate a grid of complex numbers
x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, y)
complex_grid = X + 1j * Y

# Compute phase and amplitude
phase = np.angle(complex_grid)  # Phase in radians (-π to π)
amplitude = np.abs(complex_grid)  # Magnitude

# Normalize phase to [0, 1] (map -π to π -> 0 to 1)
normalized_phase = (phase + np.pi) / (2 * np.pi)

# Normalize amplitude to [0, 1] for transparency
normalized_amplitude = amplitude / np.max(amplitude)

# Map phase to colors using the HSV colormap
colormap = plt.cm.hsv  # HSV colormap for phase
colors = colormap(normalized_phase)  # Map phase to RGB

# Set transparency (alpha) based on amplitude
colors[..., -1] = normalized_amplitude  # Set the alpha channel

# Display the result
plt.imshow(colors, extent=(-2, 2, -2, 2))
plt.title("Phase (0 to 360°) Mapped to Colors, Amplitude to Transparency")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()