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
particule_initiale_px = (500, 300) # à changer

# Paramètres de la simulation
f2 = 150  # Focale de L2
na = 0.85  # Numerical aperture
lamb = 0.405  # Wavelength in um
M_theo = 60  # Magnification of the objective


output_dir = 'runs/f2=150_lamb=405_na=0,85_Mtheo=60_Size=1um-2'

# D théorique, taille pxiel et variance théorique
#D = (1.38 * 10**-23 * 300 / (6 * np.pi * 10**(-3) * 0.5*10**-6))  # Diffusion coefficient
# D = 1.0981691 * 10**(-13) # m^2/s
# nb_steps = 50 
# duree_totale = 1
# delta_t = duree_totale/nb_steps
# variance = np.sqrt(2*D*delta_t)*10**(6) # um
pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um
# variance_px = variance / pxl  # Variance in pixels

def visionneur(frame):
    plt.figure(figsize=(10, 5))
    plt.clf() 
    plt.imshow(frame, origin='lower', cmap='gray')
    plt.title('Grille Zoomée avec Position')
    plt.colorbar()  
    plt.show()

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    return offset + amplitude * np.exp(- (a * (x - x0)**2 + b * (y - y0)**2))

def prepare_data(x, y, z):
    return (x.flatten(), y.flatten()), z.flatten()

def localisateur_gaussien(intensity_grid, maxi):
    x = np.arange(intensity_grid.shape[0])  
    y = np.arange(intensity_grid.shape[1])  
    X, Y = np.meshgrid(x, y)  

    # Préparer les données pour le fit
    (xdata, ydata), zdata = prepare_data(X, Y, intensity_grid)
    model = lmfit.Model(gaussian_2d)
    max_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)
    initial_x0 = x[max_idx[0]]  
    initial_y0 = y[max_idx[1]]  

    # Définir les paramètres du modèle
    params = model.make_params(
        amplitude=np.max(intensity_grid),
        x0=initial_x0,
        y0=initial_y0,
        sigma_x=1,
        sigma_y=1,
        offset=2
    )

    # Effectuer l'ajustement
    result = model.fit(zdata, params, xy=(xdata, ydata))

    x_position = result.params['x0'].value + maxi[0] - 24.5
    y_position = result.params['y0'].value + maxi[1] - 24.5

    return [x_position, y_position], result.params['sigma_x'].value, result.params['sigma_y'].value

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
        msd.append(np.mean(distances_squared))  
        
        # Propagation des incertitudes
        dx = delta_x[d:] + delta_x[:-d]
        dy = delta_y[d:] + delta_y[:-d]
        term_x = 2 * (diff_pairs[:, 0]**2) * (dx**2)
        term_y = 2 * (diff_pairs[:, 1]**2) * (dy**2)
        total_uncertainty = np.mean(term_x + term_y)
        uncertainties.append(np.sqrt(total_uncertainty))
    
    return np.array(msd), np.array(uncertainties)

def crop_blob(image, index=0, crop_size=50):
    max_idx = np.unravel_index(np.argmax(image), image.shape)

    grille = np.uint8((image/np.max(image))*255)
    _, binary = cv2.threshold(grille, 125, 255, cv2.THRESH_BINARY)
    structure = np.ones((3, 3), dtype=int)  
    labeled, num_features = label(binary, structure=structure)

    # Get blob centers
    blob_centers_tranposed = center_of_mass(binary, labeled, range(1, num_features + 1))
    blob_centers=[(blob_centers_tranposed[0][1],blob_centers_tranposed[0][0])]

    # Ensure the index is within bounds
    if 0 <= index < len(blob_centers):
        x, y = max_idx[1], max_idx[0]

        x_start, x_end = max(0, x - crop_size // 2), min(grille.shape[0], x + crop_size // 2)
        y_start, y_end = max(0, y - crop_size // 2), min(grille.shape[1], y + crop_size // 2)
        cropped_image = grille[y_start:y_end, x_start:x_end]

        return cropped_image, [x, y]
    else:
        print("Invalid index!")
        return None

def positionneur(vecteur_dimages):
    trajectoire = []
    video = []
    sigma_x = []
    sigma_y = []
    for image in vecteur_dimages:
        grille_zoom, maximum = crop_blob(image)
        position, inc_x, inc_y = localisateur_gaussien(grille_zoom, maximum)

        sigma_x.append(inc_x)
        sigma_y.append(inc_y)
        video.append(image)
        trajectoire.append(position)

    return np.array(trajectoire), video, np.array(sigma_x), np.array(sigma_y)

loaded_images = []
for idx in range(nb_steps):
    filename = os.path.join(output_dir, f'image_{idx+1}.csv')
    image = np.loadtxt(filename, delimiter=',', dtype=int)
    loaded_images.append(image)

positions_estimée, video_camera, delta_x, delta_y=positionneur(loaded_images)
MSDs, msd_uncertainties = calculate_msd_with_uncertainty(positions_estimée, delta_x, delta_y, nb_steps)

# Créer le graphique
plt.figure(figsize=(8, 6))

plt.scatter(range(len(MSDs)), MSDs, color='blue')
plt.xlabel('delta t')
plt.ylabel('MSD')
plt.title('MSD en fonction du temps')

plt.show()


# Graphique de régression linéaire
def linear_model(x, m, b):
    return m * x + b
y_errors = msd_uncertainties[:7]  
y_data = MSDs[:7] 
x_data = np.arange(len(y_data)) * delta_t 
params, covariance = curve_fit(linear_model, x_data, y_data, sigma=y_errors)
m_fit, b_fit = params
m_uncertainty, b_uncertainty = np.sqrt(np.diag(covariance))

print(f"Pente ajustée : {m_fit:.2f} ± {m_uncertainty:.2f}")
print(f"Ordonnée à l'origine : {b_fit:.2f} ± {b_uncertainty:.2f}")

plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', label='Données avec incertitudes', capsize=5)
plt.plot(x_data, linear_model(x_data, *params), label=f'Fit: y = {m_fit:.2f}x + {b_fit:.2f}', color='red')
plt.xlabel('Nombre dintervalles (x)')
plt.ylabel('MSD (y)')
plt.legend()
plt.title("Régression linéaire avec incertitudes propagées")
plt.show()



# Calcul de D et de la taille de la particule
Taille = 1 # um 
D_estime = (m_fit/4 )*(pxl**2) # um^2/s
D_inc_estime = (m_uncertainty/4)*(pxl**2) # um^2/s
Taille_estime = (1.38 * 10**-23 * 300 / (6 * np.pi * 10**(-3) * D_estime))
#Taille_inc_estime = Taille * D_inc_estime/(D*(10**12)) # um

print(f'D estimé = {D_estime} ± {D_inc_estime} um^2/s')
print(f'Taille estimé = {Taille_estime} ± {Taille_inc_estime} um')



# Film du déplacement de la particule avec sa trajectoire
pause_duration = 0.1
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
    frame_zoom = frame[y_min:y_max, x_min:x_max] 
    max_intensity_index = np.unravel_index(np.argmax(frame_zoom), frame_zoom.shape)
    
    max_x = max_intensity_index[1] + x_min  
    max_y = max_intensity_index[0] + y_min  

    plt.clf()  
    plt.imshow(frame_zoom, origin='lower', cmap='gray', extent=[x_position.min(), x_position.max(), y_position.min(), y_position.max()])
    plt.title('Grille Zoomée avec Position')
    plt.xlabel('Pixels en x')
    plt.ylabel('Pixels en y')
    cb = plt.colorbar()  
    cb.set_label('Nombre de photons (Intensité)')

    trajectory.append(position)
    plt.scatter(position[0], position[1], color='red', s=50) 
    if len(trajectory) > 1:  
        plt.plot([pos[0] for pos in trajectory], [pos[1] for pos in trajectory], color='red', lw=2)
    plt.pause(pause_duration)

plt.show()


