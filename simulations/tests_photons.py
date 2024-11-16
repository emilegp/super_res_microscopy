import numpy as np
from scipy.special import j1  # Bessel function of the first kind
import matplotlib.pyplot as plt

def psf_r(r, NA, wavelength):
    """Point spread function as a function of radial distance r."""
    k = 2 * np.pi * NA / wavelength
    return (2 * j1(k * r) / (k * r)) ** 2

def generate_photon_coordinates(NA, wavelength, num_photons=1):
    """Generate photon emission coordinates following the PSF."""
    # Define the maximum radius for the simulation (e.g., 3 times the Airy disk radius)
    r_max = 5 * wavelength / (2 * NA)
    r_values = np.linspace(0, r_max, 10000)  # Fine grid for PSF evaluation
    
    # Compute PSF and normalize to create a probability distribution
    psf_values = psf_r(r_values, NA, wavelength)
    psf_values[0] = 0  # Avoid division by zero at r=0
    psf_cdf = np.cumsum(psf_values * np.diff(np.append(0, r_values)))  # CDF of PSF
    psf_cdf /= psf_cdf[-1]  # Normalize to 1

    # Generate random samples for r using inverse transform sampling
    random_samples = np.random.rand(num_photons)
    sampled_r = np.interp(random_samples, psf_cdf, r_values)
    
    # Generate random theta values (uniformly distributed)
    theta = np.random.uniform(0, 2 * np.pi, num_photons)
    
    # Convert polar coordinates to Cartesian
    x = sampled_r * np.cos(theta)
    y = sampled_r * np.sin(theta)
    return x, y

# Example usage
NA = 1.4  # Numerical aperture
wavelength = 500e-9  # Wavelength in meters (e.g., 500 nm)
num_photons = 100

x_coords, y_coords = generate_photon_coordinates(NA, wavelength, num_photons)

# Plot the photon emission pattern
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, s=3, alpha=0.5)
plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.title("Simulated Photon Emission Following PSF")
plt.axis("equal")
plt.show()
