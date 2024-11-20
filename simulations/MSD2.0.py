import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function

pixel_size = 3.45 # en um, x et y
cam_width = 1440
cam_height = 1080
particule_initiale_px = (500, 300)

# Paramètres de la simulation
f2 = 0.4  # Facteur de l'objectif
na = 1.4  # Numerical aperture
lamb = 0.405  # Wavelength in um
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
    k = 2 * np.pi * na / (lamb) 

    # Airy disk PSF
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = ((2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) ** 2)
    psf[r == 0] = 1  # Treatment of singularity

    return psf

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

# Simuler les localisations
#D = (1.38 * 10**-23 * 300 / (6 * np.pi * 10**(-3) * 10**-6))  # Diffusion coefficient
D = 2.196338215 * 10**(-13) # m^2/s
nb_steps = 10000
duree_totale = 10
delta_t = duree_totale/nb_steps
variance = np.sqrt(2*D*delta_t)*10**(6) # um
pxl = pixel_size / (f2 * M_theo / 160)  # Pixel size in um
variance_px = variance / pxl  # Variance in pixels


localisations_px_cumsum = Deplacement_brownien(particule_initiale_px, variance_px, nb_steps)
MSD = MSD_cumsum(localisations_px_cumsum, nb_steps)

# Calcul de pente pour vérifier D rapidement
pente = (MSD[50]-MSD[0])/(delta_t*(50-0))
D_estime = pente/4 
D_px = D*10**12/pxl**2 #en um^2/s dans le plan en pixel^2/s

print(f'D théorique = {D_px} = {D_estime} = D estimé !!! À 10% près...')

# Tracer le MSD en fonction du temps
plt.plot(np.arange(1, len(MSD) + 1), MSD, label="MSD")
plt.xlabel('Temps (pas)')
plt.ylabel('MSD (pixels^2)')
plt.title("Déplacement quadratique moyen (MSD) avec cumsum vs Temps")
plt.grid(True)
plt.show()

# Tracer le MSD en fonction du temps
plt.plot(np.arange(1, 50 + 1), MSD[:50], label="MSD")
plt.xlabel('Temps (pas)')
plt.ylabel('MSD (pixels^2)')
plt.title("Déplacement quadratique moyen (MSD) avec cumsum vs Temps")
plt.grid(True)
plt.show()

# Estimation du coefficient de diffusion (D)
# Si le MSD suit une relation linéaire avec le temps : MSD(t) = 4Dt
# Vous pouvez ajuster une droite à ces données pour en déduire D.
