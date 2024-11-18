import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import pi
from scipy.special import j1  # Bessel function
import matplotlib.animation as animation

# Variables

num_time_steps = 10
time_step = 0.1
diff_coeff = 0

nb_particules = 1
na = 1.4
lamb = 405 # en nm, mettre en m?


# Initialisaton de la simulation

size = 100
x = np.linspace(-100, 100, size)
y = np.linspace(-100, 100, size)
X, Y = np.meshgrid(x, y)

# def PSF
def psf(x,y):
    r = np.sqrt(x**2+y**2)
    k = 2 * pi * na / lamb
    # Safely handle r = 0
    psf = np.zeros_like(r)
    nonzero_indices = r > 0
    psf[nonzero_indices] = ((2 * j1(k * r[nonzero_indices]) / (k * r[nonzero_indices])) **2)
    psf[r == 0] = 1  # Define PSF at r = 0 explicitly
    return psf



# Distribution de Poisson



# Compte de photons


# Evaluate the PSF on the grid
psf_values = psf(X, Y)

# Normalize the PSF to make it a probability distribution
psf_values /= psf_values.sum()

# Flatten the array to make it easier to sample from
psf_flat = psf_values.flatten()

# Sample a random index from the flattened PSF array based on the distribution
index = np.random.choice(len(psf_flat), p=psf_flat)

# Convert the index back to 2D coordinates (x, y)
random_x = X.flatten()[index]
random_y = Y.flatten()[index]

print(f"Random point: x = {random_x}, y = {random_y}")

# Visualize the PSF distribution
plt.imshow(psf_values, extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(label="PSF Value")
plt.title("Point Spread Function (PSF)")
plt.scatter(random_x, random_y, color='red', label=f'Random Point: ({random_x:.2f}, {random_y:.2f})')
plt.legend()
plt.show()


plt.show()


# Fit gaussien



# Déplacement par mouvement brownien



# MSD



# Régression linéaire sur MSD



# Animation
