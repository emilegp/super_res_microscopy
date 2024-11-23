import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function
from scipy.ndimage import label, center_of_mass
from scipy.optimize import curve_fit
import lmfit
import cv2
import os

pixel_size = 3.45 # en um, x et y
cam_width = 1440
cam_height = 1080
particule_initiale_px = (500, 300)

# Paramètres de la simulation
f2 = 150  # Facteur de l'objectif
na = 0.4  # Numerical aperture
lamb = 0.405  # Wavelength in um
M_theo = 20  # Magnification of the objective
poisson_lamb = 200  # Average number of photons
mean_photon_count = 1  # Mean number of photons emitted

# D théorique, taille pxiel et variance théorique
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

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    return offset + amplitude * np.exp(- (a * (x - x0)**2 + b * (y - y0)**2))

def prepare_data(x, y, z):
    return (x.flatten(), y.flatten()), z.flatten()

def localisateur_gaussien(intensity_grid):
    ##### Fit de Gaussienne #####

    # Générer les coordonnées x et y pour chaque pixel de l'image
    x = np.arange(intensity_grid.shape[0])  # Coordonnées x de chaque pixel
    y = np.arange(intensity_grid.shape[1])  # Coordonnées y de chaque pixel
    X, Y = np.meshgrid(x, y)  # Crée un maillage de coordonnées (X, Y)

    # Préparer les données pour le fit
    (xdata, ydata), zdata = prepare_data(X, Y, intensity_grid)

    # Créer le modèle Gaussien et définir les paramètres initiaux
    model = lmfit.Model(gaussian_2d)

    # Trouver les indices de l'intensité maximale
    max_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)

    # Convertir les indices en coordonnées
    initial_x0 = x[max_idx[0]]  # coordonnée x
    initial_y0 = y[max_idx[1]]  # coordonnée y

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

    return result, result.params['sigma_x'].value, result.params['sigma_y'].value

def calculate_msd_with_uncertainty(positions, delta_x, delta_y, n_steps):
    """
    Calcule les MSD avec propagation des incertitudes analytiques.
    """
    msd = []
    uncertainties = []
    for d in range(1, n_steps):
        # Différences des paires de positions séparées par d
        diff_pairs = positions[d:] - positions[:-d]
        distances_squared = np.sum(diff_pairs**2, axis=1)
        msd.append(np.mean(distances_squared))  # Moyenne des distances au carré
        
        # Propagation des incertitudes
        dx = delta_x[d:] + delta_x[:-d]
        dy = delta_y[d:] + delta_y[:-d]
        term_x = 2 * (diff_pairs[:, 0]**2) * (dx**2)
        term_y = 2 * (diff_pairs[:, 1]**2) * (dy**2)
        total_uncertainty = np.mean(term_x + term_y)
        uncertainties.append(np.sqrt(total_uncertainty))
    
    return np.array(msd), np.array(uncertainties)

def crop_blob(image, index=0, crop_size=50):
    grille = np.uint8((image/np.max(image))*255)

    # Threshold to create a binary image
    _, binary = cv2.threshold(grille, 175, 255, cv2.THRESH_BINARY)

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
    trajectoire = []
    video = []
    sigma_x = []
    sigma_y = []
    for image in vecteur_dimages:
        grille_zoom, point_de_reference  = crop_blob(image)[0], crop_blob(image)[1]
        result, inc_x, inc_y = localisateur_gaussien(grille_zoom)
        sigma_x.append(inc_x)
        sigma_y.append(inc_y)
        video.append(image)
        
        # Extraire les paramètres ajustés pour la moyenne de la gaussienne
        x_position = result.params['x0'].value + point_de_reference[0]
        y_position = result.params['y0'].value + point_de_reference[1]
        trajectoire.append([x_position, y_position])

    return np.array(trajectoire), video, np.array(sigma_x), np.array(sigma_y)

output_dir = 'runs/test'
loaded_images = []
for idx in range(nb_steps):
    filename = os.path.join(output_dir, f'image_{idx+1}.csv')
    image = np.loadtxt(filename, delimiter=',', dtype=int)
    loaded_images.append(image)
    print(f'Loaded image_{idx+1}.csv')

positions_estimée, video_camera, delta_x, delta_y=positionneur(loaded_images)
MSDs, msd_uncertainties = calculate_msd_with_uncertainty(positions_estimée, delta_x, delta_y, nb_steps)

# Créer le graphique
plt.figure(figsize=(8, 6))

plt.scatter(range(len(MSDs)), MSDs, color='blue')
plt.xlabel('delta t')
plt.ylabel('MSD')
plt.title('MSD en fonction du temps')

plt.show()



# Define the linear model (linear regression function)
def linear_model(x, m, b):
    return m * x + b
y_errors = msd_uncertainties[:7]  
y_data = MSDs[:7] 
x_data = np.arange(len(y_data))  
params, covariance = curve_fit(linear_model, x_data, y_data, sigma=y_errors)
m_fit, b_fit = params
m_uncertainty, b_uncertainty = np.sqrt(np.diag(covariance))

print(f"Pente ajustée : {m_fit:.2f} ± {m_uncertainty:.2f}")
print(f"Ordonnée à l'origine : {b_fit:.2f} ± {b_uncertainty:.2f}")

# Tracer le graphique avec les incertitudes
plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', label='Données avec incertitudes', capsize=5)
plt.plot(x_data, linear_model(x_data, *params), label=f'Fit: y = {m_fit:.2f}x + {b_fit:.2f}', color='red')
plt.xlabel('Nombre dintervalles (x)')
plt.ylabel('MSD (y)')
plt.legend()
plt.title("Régression linéaire avec incertitudes propagées")
plt.show()

# Calcul de D et de la taille de la particule
D_estime = (m_fit/4 )*(pxl**2) # um^2/s
Taille_estime = D_estime

print(f'D estimé = {D_estime} um^2/s et vrai D = {D*(10**12)} um^2/s')
print(f'Taille estimé = {2*D_estime/(D*(10**12))} um et vrai Taille = {1} um')


# Définir la durée de la pause entre chaque image (en secondes)
pause_duration = 0.5
trajectory = []
x_range = np.arange(50) - 25 
y_range = np.arange(50) - 25
x_position = x_range + positions_estimée[0][0]  
y_position = y_range + positions_estimée[0][1] 
x_min, x_max = int(np.floor(x_position.min())), int(np.ceil(x_position.max()))
y_min, y_max = int(np.floor(y_position.min())), int(np.ceil(y_position.max()))

plt.figure(figsize=(10, 5))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for frame, position in zip(video_camera, positions_estimée):
    frame_zoom = frame[x_min:x_max, y_min:y_max]  # Utilisation de slice (x_min:x_max, y_min:y_max)
    plt.clf()  # Effacer la figure précédente pour éviter l'empilement des images

    # Définir les axes X et Y de la figure actuelle comme x_position et y_position
    plt.imshow(frame_zoom, origin='lower', cmap='gray', extent=[x_position.min(), x_position.max(), y_position.min(), y_position.max()])
    plt.title('Grille Zoomée avec Position')
    plt.colorbar()  # Ajouter la colorbar

    # Ajouter la position actuelle à la trajectoire
    trajectory.append(position)
    plt.scatter(position[0], position[1], color='red', s=50)  # Position marquée en rouge

    if len(trajectory) > 1:  # Si on a plus d'une position
        plt.plot([pos[0] for pos in trajectory], [pos[1] for pos in trajectory], color='red', lw=2)

    plt.pause(pause_duration)

plt.show()


