import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function
import lmfit

pixel_size = 3.45 # en um, x et y
cam_width = 1440
cam_height = 1080
particule_initiale_px = (500, 300)

# Paramètres de la simulation
f2 = 150  # Facteur de l'objectif
na = 0.4  # Numerical aperture
lamb = 0.405  # Wavelength in um
M_theo = 20  # Magnification of the objective
poisson_lamb = 400  # Average number of photons
mean_photon_count = 5  # Mean number of photons emitted

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
    k = 2 * np.pi * na / (lamb)  # Wavelength en um

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

def zoom(grid, loc):
    return grid[int(loc[1])-10 : int(loc[1])+11, int(loc[0])-10 : int(loc[0])+11]

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
    print('début Airy')
    for _ in range(num_simulations):
        index = np.random.choice(len(psf_flat), p=psf_flat)

        random_x = X.flatten()[index]
        random_y = Y.flatten()[index]

        # Find the nearest grid indices
        grid_x_idx = np.argmin(np.abs(x - random_x))
        grid_y_idx = np.argmin(np.abs(y - random_y))
        
        # Count photon emission
        intensity_grid[grid_y_idx, grid_x_idx] += 1
    print('fin Airy')

    # Plot the original and fitted intensity grids using imshow
    plt.figure(figsize=(10, 5))

    # Original intensity grid with a cross at the mean of the Gaussian (from the fit)
    plt.imshow(intensity_grid, origin='lower', cmap='gray')
    plt.title('Original Intensity Grid')
    plt.colorbar()
    plt.show()

    return intensity_grid

def localisateur_gaussien(intensity_grid):
##### Fit de Gaussienne #####

    (xdata, ydata), zdata = prepare_data(X, Y, intensity_grid)

# Create lmfit model and initial parameters
    model = lmfit.Model(gaussian_2d)

# Find the indices of the maximum intensity
    max_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)

# Convert the indices to coordinates
    initial_x0 = x[max_idx[1]]  # x-coordinate
    initial_y0 = y[max_idx[0]]  # y-coordinate
    print('après coordonées')

    params = model.make_params(
        amplitude=np.max(intensity_grid),
        x0=initial_x0,
        y0=initial_y0,
        sigma_x=10,
        sigma_y=10,
        offset=0
    )
    print('après paramètres')

# Perform the fit
    result = model.fit(zdata, params, xy=(xdata, ydata))
    print('après fit')


    return [result, intensity_grid]


def Deplacement_brownien(particule_loc, sigma, n_steps):
    dx=np.random.normal(0,sigma, n_steps)
    dy=np.random.normal(0,sigma, n_steps)

    positions = np.cumsum(np.array([dx, dy]), axis=1).T + np.array(particule_loc)  
    return positions

def MSD_cumsum(positions, n_steps):
    # Calculer les distances pour chaque écart possible (en vectorisant le calcul)
    msd = []
    for d in range(1, n_steps):  # On commence à d=1 car la distance 0 est triviale
        # Calculer toutes les paires (i, i+d) de positions
        diff_pairs = positions[d:] - positions[:-d]  # Différences entre positions séparées par d indices
        distances_squared = np.sum(diff_pairs**2, axis=1)  # Distances au carré entre les paires
        msd.append(np.mean(distances_squared))  # Moyenne des distances au carré
    
    return np.array(msd)

# Function to crop around a specific blob by index (using zero-based index)
def crop_blob(image, blob_centers, index=0, crop_size=50):
    # Ensure the index is within bounds
    if 0 <= index < len(blob_centers):
        # Get the coordinates of the selected blob
        x, y = int(blob_centers[index][0]), int(blob_centers[index][1])

        # Define crop boundaries
        x_start, x_end = max(0, x - crop_size // 2), min(image.shape[0], x + crop_size // 2)
        y_start, y_end = max(0, y - crop_size // 2), min(image.shape[1], y + crop_size // 2)

        # Crop the image around the blob
        cropped_image = image[x_start:x_end, y_start:y_end]
        return [cropped_image, [x_start, y_start]]
    else:
        print("Invalid index!")
        return None

# Simuler les localisations
#D = (1.38 * 10**-23 * 300 / (6 * np.pi * 10**(-3) * 10**-6))  # Diffusion coefficient
D = 2.196338215 * 10**(-13) # m^2/s
nb_steps = 10000
duree_totale = 10
delta_t = duree_totale/nb_steps
variance = np.sqrt(2*D*delta_t)*10**(6) # um
pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um
variance_px = variance / pxl  # Variance in pixels


localisations_px = Deplacement_brownien(particule_initiale_px, variance_px, nb_steps)
MSDs = MSD_cumsum(localisations_px, nb_steps)






##### RUN #####

#for positions in localisations_px:
#    print(positions)

result = blob((500,300), f2, na, lamb, M_theo, poisson_lamb, mean_photon_count)[0]
print('après result')
# Plot the original and fitted intensity grids
Z_fit = result.best_fit.reshape(X.shape)

# Extract the fitted parameters for the mean of the Gaussian
x0_fit = result.params['x0'].value
y0_fit = result.params['y0'].value

print(f"Point moyen = ({x0_fit:.2f}, {y0_fit:.2f}) pixels")

