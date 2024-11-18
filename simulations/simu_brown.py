import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function
import lmfit

# Variables
num_time_steps = 10
time_step = 0.1
diff_coeff = 0
nb_particules = 1
na = 0.4
lamb = 405  # in nm, convert to meters in calculations if needed
poisson_lambda = 800

# Camera specifications
pixel_size = 3.45  # Pixel size in micrometers
width = 1440  # Number of pixels horizontally
height = 1080  # Number of pixels vertically

# Define the pixel coordinates in micrometers
x = np.linspace(pixel_size / 2, (width - 0.5) * pixel_size, width)
y = np.linspace(pixel_size / 2, (height - 0.5) * pixel_size, height)
X, Y = np.meshgrid(x, y)

# Particle positions in micrometers (relative to the grid center)
particle_positions = [(1000, 2000), (-1500, -1000), (0, 0)]  # Example positions in micrometers

# PSF, singularity treated at r = 0
def psf(x, y):
    r = np.sqrt(x**2 + y**2)
    k = 2 * pi * na / (lamb * 1e-3)  # Convert lambda to micrometers
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = ((2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) ** 2)
    psf[r == 0] = 1
    return psf

# Initialize intensity grid for photon counts
intensity_grid = np.zeros_like(X)

# Simulate photon counts for each particle
for particle_x, particle_y in particle_positions[:1]:  # Use only the first particle for this example
    # Shift the grid to center the PSF at the particle position
    shifted_X = X - particle_x
    shifted_Y = Y - particle_y
    psf_values_shifted = psf(shifted_X, shifted_Y)
    psf_values_shifted /= psf_values_shifted.sum()  # Normalize to 1
    
    # Flatten for sampling
    psf_flat = psf_values_shifted.flatten()
    
    # Simulate photon count based on Poisson distribution
    num_simulations = np.random.poisson(poisson_lambda)
    for _ in range(num_simulations):
        index = np.random.choice(len(psf_flat), p=psf_flat)
        
        random_x = X.flatten()[index]
        random_y = Y.flatten()[index]
        
        # Find the nearest grid indices
        grid_x_idx = np.argmin(np.abs(x - random_x))
        grid_y_idx = np.argmin(np.abs(y - random_y))
        
        # Count photon emission
        intensity_grid[grid_y_idx, grid_x_idx] += 1

# Define 2D Gaussian function for fitting
def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    return offset + amplitude * np.exp(- (a * (x - x0)**2 + b * (y - y0)**2))

# Prepare data for fitting
def prepare_data(x, y, z):
    return (x.flatten(), y.flatten()), z.flatten()

(xdata, ydata), zdata = prepare_data(X, Y, intensity_grid)

# Create lmfit model and initial parameters
model = lmfit.Model(gaussian_2d)

# Find the indices of the maximum intensity
max_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)

# Convert the indices to coordinates
initial_x0 = x[max_idx[1]]  # x-coordinate
initial_y0 = y[max_idx[0]]  # y-coordinate

params = model.make_params(
    amplitude=np.max(intensity_grid),
    x0=initial_x0,
    y0=initial_y0,
    sigma_x=10,
    sigma_y=10,
    offset=0
)

# Perform the fit
result = model.fit(zdata, params, xy=(xdata, ydata))

# Print the fit result
print(result.fit_report())

# Plot the original and fitted intensity grids
Z_fit = result.best_fit.reshape(X.shape)

# Extract the fitted parameters for the mean of the Gaussian
x0_fit = result.params['x0'].value
y0_fit = result.params['y0'].value

print(f"Point moyen = ({x0_fit:.2f}, {y0_fit:.2f}) µm")

# Plot the original and fitted intensity grids using imshow
plt.figure(figsize=(10, 5))

# Original intensity grid with a cross at the mean of the Gaussian (from the fit)
plt.subplot(1, 2, 1)
plt.imshow(intensity_grid, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title('Original Intensity Grid')
plt.colorbar()
plt.plot(x0_fit, y0_fit, 'rx', markersize=10)  # Plot cross at the mean of the Gaussian

# Fitted intensity grid
plt.subplot(1, 2, 2)
plt.imshow(Z_fit, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
plt.title('Fitted Intensity Grid')
plt.colorbar()

plt.tight_layout()
plt.show()
