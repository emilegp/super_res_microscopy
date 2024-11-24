import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function
import os

pixel_size = 3.45 # en um, x et y
cam_width = 1440
cam_height = 1080
particule_initiale_px = (500, 300)

# Paramètres de la simulation
f2 = 150  # Facteur de l'objectif
na = 0.85  # Numerical aperture
lamb = 0.405  # Wavelength in um
M_theo = 60  # Magnification of the objective
poisson_lamb = 400  # Average number of photons
mean_photon_count = 2  # Mean number of photons emitted

output_dir = 'runs/f2=150_lamb=405_na=0,85_Mtheo=60_Size=1um-2'

# Simuler les localisations
#D = (1.38 * 10**-23 * 300 / (6 * np.pi * 10**(-3) * 0.5*10**-6))  # Diffusion coefficient
D = 1.0981691 * 10**(-13) # m^2/s
nb_steps = 50
duree_totale = 1
delta_t = duree_totale/nb_steps
variance = np.sqrt(2*D*delta_t)*10**(6) # um
pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um
variance_px = variance / pxl  # Variance in pixels

x = np.arange(cam_width)
y = np.arange(cam_height)
X, Y = np.meshgrid(x, y)

def visionneur(video_camera, index):
    plt.figure(figsize=(10, 5))
    frame_zoom = video_camera[index]  # Utilisation de slice (x_min:x_max, y_min:y_max)
    plt.clf()  # Effacer la figure précédente pour éviter l'empilement des images
    plt.imshow(frame_zoom, origin='lower', cmap='gray')
    plt.title('Grille Zoomée avec Position')
    plt.colorbar()  # Ajouter la colorbar
    plt.show()

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

def cameraman_god(particle_loc, f2, na, lamb, M_theo, poisson_lamb, mean_photon_count):
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

    # # Plot the original and fitted intensity grids using imshow
    # plt.figure(figsize=(10, 5))

    # # Original intensity grid with a cross at the mean of the Gaussian (from the fit)
    # plt.imshow(intensity_grid, origin='lower', cmap='gray')
    # plt.title('Original Intensity Grid')
    # plt.colorbar()
    # plt.show()

    return intensity_grid

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

##### Partie God #####
localisations_px = Deplacement_brownien(particule_initiale_px, variance_px, nb_steps)
grille = cameraman_god((500,300), f2, na, lamb, M_theo, poisson_lamb, mean_photon_count)
MSDs_god = MSD_cumsum(localisations_px, nb_steps)

images_progression=[]
for position_au_temps_t in localisations_px:
    frame=cameraman_god((position_au_temps_t[0], position_au_temps_t[1]), f2, na, lamb, M_theo, poisson_lamb, mean_photon_count)
    images_progression.append(frame)
    print('image obtenue')

# Dossier de sauvegarde
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder chaque image sous forme de fichier CSV
for idx, image in enumerate(images_progression):
    filename = os.path.join(output_dir, f'image_{idx+1}.csv')
    np.savetxt(filename, image, delimiter=',', fmt='%d')  # Sauvegarde en format entier
    print(f'Saved image_{idx+1}.csv')


