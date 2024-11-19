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
na = 1.4
lamb = 405  # in nm, convert to meters in calculations if needed
poisson_lambda = 400

# Initialisation de la simulation
x = np.linspace(0, 1439, 1440)
y = np.linspace(0, 1079, 1080)
X, Y = np.meshgrid(x, y)

# PSF, singularité traitée en r = 0
def psf(x, y):
    r = np.sqrt(x**2 + y**2)
    k = 2 * pi * na / lamb
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = ((2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) ** 2)
    psf[r == 0] = 1
    return psf

# Génération de la PSF partout
psf_values = psf(X, Y)
psf_values /= psf_values.sum()  # somme des probas à 1

psf_flat = psf_values.flatten()

# Initialize intensity grid for photon counts
intensity_grid = np.zeros_like(psf_values)

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
params = model.make_params(amplitude=np.max(intensity_grid), x0=0, y0=0, sigma_x=10, sigma_y=10, offset=0)

# Perform the fit
result = model.fit(zdata, params, xy=(xdata, ydata))

# Print the fit result
print(result.fit_report())

# Plot the original and fitted intensity grids
Z_fit = result.best_fit.reshape(X.shape)

# Extract the fitted parameters for the mean of the Gaussian
x0_fit = result.params['x0'].value
y0_fit = result.params['y0'].value

print(f"point moyen = ({x0_fit}, {y0_fit})")

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


# Déplacement par mouvement brownien



# MSD



# Régression linéaire sur MSD



# Animation
