import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function
import lmfit


def psf(xx, yy, particle_pos, na=0.4, lamb=0.5, effective_pixel_size=0.1725):
    """
    Compute the PSF intensity based on pixel coordinates.
    
    Parameters:
    xx, yy : 2D arrays of pixel coordinates
    particle_pos : tuple (px, py) of particle's center in pixel coordinates
    na : Numerical aperture of the objective
    lamb : Wavelength in micrometers
    effective_pixel_size : Effective size of one pixel in the sample plane (μm)
    """
    # Convert pixel coordinates to real-world distances
    px, py = particle_pos
    x_um = (xx - px) * effective_pixel_size
    y_um = (yy - py) * effective_pixel_size
    r = np.sqrt(x_um**2 + y_um**2)  # Distance in μm

    # Wave vector
    k = 2 * np.pi * na / (lamb * 1e-3)  # Wavelength in μm

    # Airy disk calculation
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = (2 * np.sinc(k * r[nonzero_indices])) ** 2
    psf[r == 0] = 1  # Singularity at the center

    return psf

# Camera dimensions
camera_width = 1440
camera_height = 1080
x = np.arange(camera_width)
y = np.arange(camera_height)
xx, yy = np.meshgrid(x, y)
effective_pixel_size=0.1725
# Particle position in pixel coordinates
particle_position_pixel = (720, 540)  # Center of the grid

# Generate PSF
airy_disk = psf(xx, yy, particle_position_pixel, na=0.4, lamb=405, effective_pixel_size=effective_pixel_size)

# Visualize
plt.imshow(airy_disk, cmap='hot', origin='lower')
plt.colorbar(label='Intensity')
plt.title("Airy Disk PSF with Real Dimensions")
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.show()
