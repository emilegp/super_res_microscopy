import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function
from scipy.ndimage import label, center_of_mass
from scipy.optimize import curve_fit
import lmfit
import cv2
import os
from PIL import Image

# Paramètres du set-up
pixel_size = 3.45 # en um, x et y
cam_width = 1440
cam_height = 1080

f2 = 150  # Focale de L2
na = 0.85  # Numerical aperture
lamb = 0.405  # Wavelength in um
M_theo = 60  # Magnification of the objective

pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um

def visionneur(frame):
    # if frame[0][0][0] is not None:
    #     frame = frame[0]
    plt.figure(figsize=(10, 5))
    plt.clf() 
    plt.imshow(frame, origin='lower', cmap='gray')
    plt.title('Grille Zoomée avec Position')
    plt.colorbar()  
    plt.show()

def visionneur_video(video_camera, index):
    plt.figure(figsize=(10, 5))
    frame_zoom = video_camera[index]  
    plt.clf()  
    plt.imshow(frame_zoom, origin='lower', cmap='gray')
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

def crop_blob(image, index=20, crop_size=50):
    visionneur(image)

    grille = np.uint8((image/np.max(image))*255)
    _, binary = cv2.threshold(grille, 125, 255, cv2.THRESH_BINARY)
    structure = np.ones((3, 3), dtype=int)  
    labeled, num_features = label(binary, structure=structure)

    # Get blob centers
    blob_centers = center_of_mass(binary, labeled, range(1, num_features + 1))
    # blob_centers=[]
    # for i in range(len(blob_centers_tranposed)):
    #     blob_centers.append((blob_centers_tranposed[i][1],blob_centers_tranposed[i][0]))

    # Clone the original image to draw circles and labels on it
    image_with_circles = cv2.cvtColor(grille, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color circles

    # Draw circles and labels around each blob center
    for i, center in enumerate(blob_centers):
        x, y = int(center[0]), int(center[1])
        radius = 10  # Set a fixed radius or calculate dynamically
        cv2.circle(image_with_circles, (y, x), radius, (0, 0, 255), 2)  # Red circle
        label_text = str(i + 1)  # Label as the index of the blob
        cv2.putText(image_with_circles, label_text, (y + 15, x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with circles and labels using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))
    plt.title("Blobs with Circles and Labels")
    plt.axis('off')
    plt.show()


    # Ensure the index is within bounds
    if 0 <= index < len(blob_centers):
        # Get the coordinates of the selected blob
        x, y = int(blob_centers[index][0]), int(blob_centers[index][1])

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

def convertisseur(fichier, type):
    if type == 'photo':
        fichier.seek(0)
        frame_data = fichier.convert("L")  # Convert to grayscale if needed
        image = np.array(frame_data)

        # visionneur(image)

        return image
    else:
        nb_frames = fichier.n_frames

        image_array=[]
        for frame in range(nb_frames):
            fichier.seek(frame)
            image = np.array(fichier.convert("L"))
            image_array.append(image)

            # # Display the frame
            # plt.imshow(image, cmap="gray")
            # plt.xlabel('Pixels en x')
            # plt.ylabel('Pixels en y')
            # plt.title(f"Frame {frame}")
            # plt.show()
        return image_array

# Open the TIFF file
tiff_file_fond = "simulations\\microscopy_image.tiff"
fond = Image.open(tiff_file_fond)
array_fond=convertisseur(fond, 'photo')

nb_steps = fond.n_frames
duree_totale = 1 #Temps en secondes
delta_t = duree_totale/nb_steps


positions_estimée_micro, video_camera_micro, delta_x_micro, delta_y_micro=positionneur([array_fond])
#Afficher les infos
print(f'position:{positions_estimée_micro}')



# # Film du déplacement de la particule avec sa trajectoire
# pause_duration = 0.1
# trajectory = []
# x_range = np.arange(50) - 25 
# y_range = np.arange(50) - 25
# x_position = x_range + positions_estimée_micro[0][0]  
# y_position = y_range + positions_estimée_micro[0][1] 
# x_min, x_max = int(np.floor(x_position.min())), int(np.ceil(x_position.max()))
# y_min, y_max = int(np.floor(y_position.min())), int(np.ceil(y_position.max()))

# plt.figure(figsize=(10, 5))
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# #Problème d'affichage causés par la variance dans le suivage de particule (pas la position de toujours la meme particule)

# for indice in range(len(positions_estimée_micro)):
#     #frame_zoom = array_fond[indice][y_min:y_max, x_min:x_max] 
#     frame_zoom = array_fond[indice] 
    
#     max_intensity_index = np.unravel_index(np.argmax(frame_zoom), frame_zoom.shape)
    
#     max_x = max_intensity_index[1] + x_min  
#     max_y = max_intensity_index[0] + y_min  

#     plt.clf()  
#     plt.imshow(frame_zoom, origin='lower', cmap='gray', extent=[x_position.min(), x_position.max(), y_position.min(), y_position.max()])
#     plt.title('Grille Zoomée avec Position')
#     plt.xlabel('Pixels en x')
#     plt.ylabel('Pixels en y')
#     cb = plt.colorbar()  
#     cb.set_label('Nombre de photons (Intensité)')

#     trajectory.append(positions_estimée_micro[indice])
# #    plt.scatter(positions_estimée_micro[indice][0], positions_estimée_micro[indice][1], color='red', s=50) 
# #    if len(trajectory) > 1:  
# #        plt.plot([pos[0] for pos in trajectory], [pos[1] for pos in trajectory], color='red', lw=2)
#     plt.pause(pause_duration)

# plt.show()
