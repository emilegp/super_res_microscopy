import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function
import lmfit

pixel_size = 3.45 # en um, x et y
cam_width = 1440
cam_height = 1080

x = np.arange(cam_width)
y = np.arange(cam_height)
X, Y = np.meshgrid(x, y)

def psf(xx, yy, particle_pos, na, lamb, effective_pixel_size):
    
    # s'assurer que tout est en um
    px, py = particle_pos
    x_um = (xx - px) * effective_pixel_size
    y_um = (yy - py) * effective_pixel_size
    r = np.sqrt(x_um**2 + y_um**2)  # Distance en um

    # Wave vector
    k = 2 * np.pi * na / (lamb * 1e-3)  # Wavelength en um

    # Airy disk
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = ((2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) ** 2)
    psf[r == 0] = 1  # traitement de la singularite

    return psf

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    return offset + amplitude * np.exp(- (a * (x - x0)**2 + b * (y - y0)**2))

def prepare_data(x, y, z):
    return (x.flatten(), y.flatten()), z.flatten()

def blob(particle_loc, f2, na, lamb, M_theo, poisson_lamb, mean_photon_count):
    # Magnification réelle
    M = (f2*M_theo)/160

    pxl = pixel_size/M # en um

##### grid avec bruit de poisson #####
    x = np.arange(cam_width)
    y = np.arange(cam_height)
    X, Y = np.meshgrid(x, y)
    intensity_grid = np.random.poisson(mean_photon_count, (cam_height, cam_width))

##### Emission de photons #####
    psf_values = psf(X, Y, particle_loc, na, lamb, pxl)

# somme des probas à 1
    psf_values /= psf_values.sum()
    psf_flat = psf_values.flatten()

# Simulate photon count based on Poisson distribution
    num_simulations = np.random.poisson(poisson_lamb)
    for _ in range(num_simulations):
        index = np.random.choice(len(psf_flat), p=psf_flat)
        
        random_x = X.flatten()[index]
        random_y = Y.flatten()[index]
        
        # Find the nearest grid indices
        grid_x_idx = np.argmin(np.abs(x - random_x))
        grid_y_idx = np.argmin(np.abs(y - random_y))
        
        # Count photon emission
        intensity_grid[grid_y_idx, grid_x_idx] += 1

##### Fit de Gaussienne #####

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


    return [result, intensity_grid]








##### RUN #####

result = blob((500,500), 150, 0.4, 405, 20, 400, 5)[0]

# Plot the original and fitted intensity grids
Z_fit = result.best_fit.reshape(X.shape)

# Extract the fitted parameters for the mean of the Gaussian
x0_fit = result.params['x0'].value
y0_fit = result.params['y0'].value

print(f"Point moyen = ({x0_fit:.2f}, {y0_fit:.2f}) pixels")

