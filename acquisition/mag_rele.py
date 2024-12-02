import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2

# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    return offset + amplitude * np.exp(- (a * (x - x0)**2 + b * (y - y0)**2))

# Load the image (modify if using OpenCV or Pillow)
image = cv2.imread("acquisition\\magnification_rele_crop.tif", cv2.IMREAD_GRAYSCALE)


# Select half of the image (e.g., left half)
height, width = image.shape
half_image_left = image[:, :width // 2]  # Left half
half_image_right = image[:, width // 2:]

# Generate x and y coordinates
x1 = np.arange(half_image_left.shape[1])
y1 = np.arange(half_image_left.shape[0])
x1, y1 = np.meshgrid(x1, y1)

x2 = np.arange(half_image_right.shape[1])
y2 = np.arange(half_image_right.shape[0])
x2, y2 = np.meshgrid(x2, y2)

# Flatten the data for fitting
xy1 = np.vstack((x1.ravel(), y1.ravel()))
xy2 = np.vstack((x2.ravel(), y2.ravel()))
z1 = half_image_left.ravel()
z2 = half_image_right.ravel()

# Initial guess for the fit parameters
initial_guess1 = [np.max(half_image_left), width // 4, height // 2, 5, 5, np.min(half_image_left)]
initial_guess2 = [np.max(half_image_right), width // 4, height // 2, 5, 5, np.min(half_image_right)]


# Perform the fit
popt1, pcov1 = curve_fit(gaussian_2d, xy1, z1, p0=initial_guess1)
popt2, pcov2 = curve_fit(gaussian_2d, xy2, z2, p0=initial_guess2)

# Extract the fitted parameters
amp1, x01, y01, sigma_x1, sigma_y1, offset1 = popt1
amp2, x02, y02, sigma_x2, sigma_y2, offset2 = popt2

print('centre premier point:',x01,y01,'pixels')
print('centre deuxième point:',x02,y02,'pixels')

d = (width//2+x02-x01)*3.45e-6
m = d/0.01e-3

print(m)




