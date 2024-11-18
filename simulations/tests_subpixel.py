
import numpy as np
import matplotlib.pyplot as plt

# Create the pixel grid
num_pixels = 10
pixel_size = 1.0  # Each pixel spans 1x1 unit
x = np.arange(0, num_pixels + 1) * pixel_size
y = np.arange(0, num_pixels + 1) * pixel_size
X, Y = np.meshgrid(x, y)

# Subpixel particle position
particle_position = [4.7, 6.3]  # Example position (subpixel precision)

# Plot the grid
plt.figure(figsize=(6, 6))
plt.plot(X, Y, color="black", lw=0.5)  # Grid lines
plt.plot(X.T, Y.T, color="black", lw=0.5)

# Highlight the pixels
for i in range(num_pixels):
    for j in range(num_pixels):
        plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=None, edgecolor='black', lw=0.2))

# Plot the particle position
plt.scatter(*particle_position, color="red", label="Particle (Subpixel Precision)")
plt.title("Subpixel Position on Pixelated Grid")
plt.xlabel("X (units)")
plt.ylabel("Y (units)")
plt.legend()
plt.xlim(0, num_pixels * pixel_size)
plt.ylim(0, num_pixels * pixel_size)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
