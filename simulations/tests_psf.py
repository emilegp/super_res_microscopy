import numpy as np
from scipy.special import j1  # Bessel function of the first kind
import matplotlib.pyplot as plt

def psf_r(r, NA, wavelength):
    """Point spread function as a function of radial distance r."""
    k = 2 * np.pi * NA / wavelength
    # Handle the division by zero at r=0
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = (2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) ** 2
    return psf

def psf_heatmap(NA, wavelength, grid_size=500, extent=1e-6):
    """Generate a heatmap of the PSF distribution."""
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)  # Radial distance from the origin
    
    # Compute PSF values
    psf_values = psf_r(R, NA, wavelength)
    return x, y, psf_values

# Parameters
NA = 1.4  # Numerical aperture
wavelength = 500e-9  # Wavelength in meters (e.g., 500 nm)
grid_size = 500  # Resolution of the grid
extent = 3 * wavelength / (2 * NA)  # Extent of the heatmap in meters

# Generate PSF heatmap
x, y, psf_values = psf_heatmap(NA, wavelength, grid_size, extent)

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(
    psf_values,
    cmap="hot",
    origin="lower",
)
plt.colorbar(label="PSF Intensity")
plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.title("Heatmap of PSF Distribution")
plt.axis("equal")
plt.show()
