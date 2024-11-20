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
f2 = 0.4  # Facteur de l'objectif
na = 1.4  # Numerical aperture
lamb = 405  # Wavelength in nm
M_theo = 150  # Magnification of the objective
poisson_lamb = 400  # Average number of photons
mean_photon_count = 5  # Mean number of photons emitted

# Grille de coordonnées X, Y
x = np.arange(cam_width)
y = np.arange(cam_height)
X, Y = np.meshgrid(x, y)

def psf(xx, yy, particle_pos, na, lamb, effective_pixel_size):
    px, py = particle_pos
    x_um = (xx - px) * effective_pixel_size
    y_um = (yy - py) * effective_pixel_size
    r = np.sqrt(x_um**2 + y_um**2)

    # Wavelength to wavevector k
    k = 2 * np.pi * na / (lamb * 1e-3)  # Wavelength in um

    # Airy disk PSF
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = ((2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) ** 2)
    psf[r == 0] = 1  # Treatment of singularity

    return psf

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    return offset + amplitude * np.exp(- (a * (x - x0)**2 + b * (y - y0)**2))

def prepare_data(x, y, z):
    return (x.flatten(), y.flatten()), z.flatten()

def brown(particule_loc, sigma, n_steps=10000, intervalle=1000):
    positions = [particule_loc]
    
    for step in range(1, n_steps + 1):
        dx = np.random.normal(0, sigma)
        dy = np.random.normal(0, sigma)
        
        new_position = (positions[-1][0] + dx, positions[-1][1] + dy)
        
        if step % intervalle == 0:
            positions.append(new_position)
    
    return positions

# Simuler les localisations
variance_px = 1000000000000*np.sqrt(2 * 1.38 * 10**-23 * 300 / (6 * np.pi * 1 * 1))  # Diffusion coefficient
pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um
variance = variance_px / pxl  # Variance in pixels
intervalle = 1000

localisations_px = brown(particule_initiale_px, variance, 10000, intervalle)

# Extraire les coordonnées x et y
x_vals = [pos[0] for pos in localisations_px]
y_vals = [pos[1] for pos in localisations_px]

# Tracer les positions du mouvement brownien
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', markersize=4)
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title(f'Mouvement brownien (Positions tous les 1000 pas)')
plt.grid(True)
plt.show()

def zoom(grid, loc):
    return grid[int(loc[1])-10 : int(loc[1])+11, int(loc[0])-10 : int(loc[0])+11]

# Fonction pour simuler un blob
def blob(particle_loc, f2, na, lamb, M_theo, poisson_lamb, mean_photon_count):
    M = (f2 * M_theo) / 160  # Réduction par le facteur de l'objectif
    pxl = pixel_size / M  # Taille du pixel en µm

    # Créer une grille avec un bruit de Poisson
    intensity_grid = np.random.poisson(mean_photon_count, (cam_height, cam_width))

    # Calculer la PSF pour la position donnée de la particule
    psf_values = psf(X, Y, particle_loc, na, lamb, pxl)

    # Normalisation de la PSF
    psf_values /= psf_values.sum()
    psf_flat = psf_values.flatten()

    # Simuler le comptage de photons
    num_simulations = np.random.poisson(poisson_lamb)
    for _ in range(num_simulations):
        index = np.random.choice(len(psf_flat), p=psf_flat)
        random_x = X.flatten()[index]
        random_y = Y.flatten()[index]
        
        # Trouver l'indice le plus proche dans la grille
        grid_x_idx = np.argmin(np.abs(x - random_x))
        grid_y_idx = np.argmin(np.abs(y - random_y))
        
        # Ajouter un photon dans la grille d'intensité
        intensity_grid[grid_y_idx, grid_x_idx] += 1

#     ##### Fit de Gaussienne #####

#     (xdata, ydata), zdata = prepare_data(X, Y, intensity_grid)

# # Create lmfit model and initial parameters
#     model = lmfit.Model(gaussian_2d)

# # Find the indices of the maximum intensity
#     max_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)

# # Convert the indices to coordinates
#     initial_x0 = x[max_idx[1]]  # x-coordinate
#     initial_y0 = y[max_idx[0]]  # y-coordinate

#     params = model.make_params(
#         amplitude=np.max(intensity_grid),
#         x0=initial_x0,
#         y0=initial_y0,
#         sigma_x=10,
#         sigma_y=10,
#         offset=0
#     )

# # Perform the fit
#     result = model.fit(zdata, params, xy=(xdata, ydata))


    return intensity_grid

print(localisations_px)


# Calcul du MSD
def calculate_msd(localisations_px):
    n = len(localisations_px)
    msd = []
    
    # On parcourt les positions de la particule
    for t in range(1, n):
        dx = localisations_px[t][0] - localisations_px[0][0]
        dy = localisations_px[t][1] - localisations_px[0][1]
        msd.append(dx**2 + dy**2)  # Calcul du carré du déplacement
        
    return np.array(msd)

# Calcul du MSD pour la simulation donnée
msd_values = calculate_msd(localisations_px)

# Tracer le MSD en fonction du temps
plt.plot(np.arange(1, len(msd_values) + 1), msd_values, label="MSD")
plt.xlabel('Temps (pas)')
plt.ylabel('MSD (pixels^2)')
plt.title("Déplacement quadratique moyen (MSD) vs Temps")
plt.grid(True)
plt.show()

# Estimation du coefficient de diffusion (D)
# Si le MSD suit une relation linéaire avec le temps : MSD(t) = 4Dt
# Vous pouvez ajuster une droite à ces données pour en déduire D.
