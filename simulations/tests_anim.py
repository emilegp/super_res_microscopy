import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function
import matplotlib.animation as animation
# Parameters
num_frames = 100
time_step = 0.1  # Time interval between steps
D = 1.0  # Diffusion coefficient

# Simulate Brownian motion
np.random.seed(42)  # For reproducibility
displacements = np.sqrt(2 * D * time_step) * np.random.randn(num_frames, 2)  # Random x, y displacements
positions = np.cumsum(displacements, axis=0)  # Cumulative sum gives position


def airy_disk(x, y, cx, cy, k=5, I0=1):
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    epsilon = 1e-10  # To avoid division by zero
    k_r = k * r + epsilon
    return I0 * (2 * j1(k_r) / k_r)**2


# Grid setup
size = 200
extent = 20
x = np.linspace(-extent, extent, size)
y = np.linspace(-extent, extent, size)
X, Y = np.meshgrid(x, y)

# Create frames
frames = []
for cx, cy in positions:
    frame = airy_disk(X, Y, cx, cy)
    frames.append(frame)


fig, ax = plt.subplots()
ax.set_title("Particle with Airy Disk (Brownian Motion)")
ax.set_xlim(-extent, extent)
ax.set_ylim(-extent, extent)
ax.set_aspect('equal')

# Initialize plot
img = ax.imshow(frames[0], extent=(-extent, extent, -extent, extent), cmap='inferno', animated=True)
cb = plt.colorbar(img, ax=ax, label='Intensity')

# Update function
def update(frame):
    img.set_array(frame)
    return img,

# Animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
ani.save("particle_brownian_motion.mp4", fps=30, writer="")
plt.show()
