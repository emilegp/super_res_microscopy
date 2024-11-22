import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function
from scipy.ndimage import label, center_of_mass
import lmfit
import cv2

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

    # # Plot the original and fitted intensity grids using imshow
    # plt.figure(figsize=(10, 5))

    # # Original intensity grid with a cross at the mean of the Gaussian (from the fit)
    # plt.imshow(intensity_grid, origin='lower', cmap='gray')
    # plt.title('Original Intensity Grid')
    # plt.colorbar()
    # plt.show()

    return intensity_grid

def localisateur_gaussien(intensity_grid):
    ##### Fit de Gaussienne #####

    # Générer les coordonnées x et y pour chaque pixel de l'image
    x = np.arange(intensity_grid.shape[1])  # Coordonnées x de chaque pixel
    y = np.arange(intensity_grid.shape[0])  # Coordonnées y de chaque pixel
    X, Y = np.meshgrid(x, y)  # Crée un maillage de coordonnées (X, Y)

    # Préparer les données pour le fit
    (xdata, ydata), zdata = prepare_data(X, Y, intensity_grid)

    # Créer le modèle Gaussien et définir les paramètres initiaux
    model = lmfit.Model(gaussian_2d)

    # Trouver les indices de l'intensité maximale
    max_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)

    # Convertir les indices en coordonnées
    initial_x0 = x[max_idx[1]]  # coordonnée x
    initial_y0 = y[max_idx[0]]  # coordonnée y

    # Définir les paramètres du modèle
    params = model.make_params(
        amplitude=np.max(intensity_grid),
        x0=initial_x0,
        y0=initial_y0,
        sigma_x=10,
        sigma_y=10,
        offset=0
    )

    # Effectuer l'ajustement
    result = model.fit(zdata, params, xy=(xdata, ydata))

    return result

def Deplacement_brownien(particule_loc, sigma, n_steps):
    dx=np.random.normal(0,sigma, n_steps)
    dy=np.random.normal(0,sigma, n_steps)

    positions = np.cumsum(np.array([dx, dy]), axis=1).T + np.array(particule_loc)  
    print(positions)
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
def crop_blob(image, index=0, crop_size=50):
    grille = np.uint8((image/np.max(image))*255)

    # Threshold to create a binary image
    _, binary = cv2.threshold(grille, 127, 255, cv2.THRESH_BINARY)

    # Detect blobs using connected-component analysis
    structure = np.ones((3, 3), dtype=int)  # Define connectivity
    labeled, num_features = label(binary, structure=structure)

    # Get blob centers
    blob_centers = center_of_mass(binary, labeled, range(1, num_features + 1))

    # Ensure the index is within bounds
    if 0 <= index < len(blob_centers):
        # Get the coordinates of the selected blob
        x, y = int(blob_centers[index][0]), int(blob_centers[index][1])

        # Define crop boundaries
        x_start, x_end = max(0, x - crop_size // 2), min(grille.shape[0], x + crop_size // 2)
        y_start, y_end = max(0, y - crop_size // 2), min(grille.shape[1], y + crop_size // 2)

        # Crop the image around the blob
        cropped_image = grille[x_start:x_end, y_start:y_end]
        return [cropped_image, [x_start, y_start]] #OU position centrale possiblement
    else:
        print("Invalid index!")
        return None

def positionneur(vecteur_dimages):
    trajectoire=[]
    for image in vecteur_dimages:
        grille_zoom, point_de_reference  = crop_blob(image)[0], crop_blob(image)[1]
        result = localisateur_gaussien(grille_zoom)

        #Redimensionner le meilleur ajustement à la forme correcte de grille_zoom
        Z_fit = result.best_fit.reshape(grille_zoom.shape)  # Utiliser la forme de grille_zoom

        # Extraire les paramètres ajustés pour la moyenne de la gaussienne
        x_position = result.params['x0'].value + point_de_reference[0]
        y_position = result.params['y0'].value + point_de_reference[1]
        trajectoire.append([x_position, y_position])

    return np.array(trajectoire)

# Simuler les localisations
#D = (1.38 * 10**-23 * 300 / (6 * np.pi * 10**(-3) * 10**-6))  # Diffusion coefficient
D = 2.196338215 * 10**(-13) # m^2/s
nb_steps = 50
duree_totale = 1
delta_t = duree_totale/nb_steps
variance = np.sqrt(2*D*delta_t)*10**(6) # um
pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um
variance_px = variance / pxl  # Variance in pixels

##### Partie God #####
localisations_px = Deplacement_brownien(particule_initiale_px, variance_px, nb_steps)
grille = cameraman_god((500,300), f2, na, lamb, M_theo, poisson_lamb, mean_photon_count)
MSDs_god = MSD_cumsum(localisations_px, nb_steps)

##### Partie Nous #####
images_progression=[]
for position_au_temps_t in localisations_px:
    images_progression.append(cameraman_god((position_au_temps_t[0], position_au_temps_t[1]), f2, na, lamb, M_theo, poisson_lamb, mean_photon_count))

positions_estimée=positionneur(images_progression)
MSDs = MSD_cumsum(positions_estimée, nb_steps)

## Débogage
print(f'Pos_god: {localisations_px}')
print(f'Pos_nous: {positions_estimée}')
print(f'Msd_god est = {MSDs_god}')
print(f'Msd_nous est ={MSDs}')

# Créer le graphique
plt.figure(figsize=(8, 6))

plt.scatter(range(len(MSDs)), MSDs, color='blue')
#plt.scatter(x2, y2, color='red', label='Vecteur 2', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Affichage de deux vecteurs de positions')

plt.show()

##### RUN #####

#for positions in localisations_px:
#    print(positions)

# # Affichage des résultats ajustés et originaux
# plt.figure(figsize=(10, 5))

# # Affichage de la grille zoomée
# plt.subplot(1, 2, 1)
# plt.imshow(grille_zoom, origin='lower', cmap='gray')
# plt.title('Grille Zoomée')
# plt.colorbar()

# # Affichage de l'ajustement gaussien
# plt.subplot(1, 2, 2)
# plt.imshow(Z_fit, origin='lower', cmap='gray')
# plt.title('Ajustement Gaussien')
# plt.colorbar()

# plt.show()

# # Affichage de la position dans la grille initiale
# print(f"Position du blob dans la grille initiale : ({x_blob_initial:.2f}, {y_blob_initial:.2f}) pixels")



