import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('Python_Image_Failures/sim1.jpg')

# Convert from BGR to RGB for displaying with matplotlib
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate Speckle Noise
mean = 0
stddev = 0.5  # Adjust this value to control the noise level
gauss = np.random.normal(mean, stddev, img.shape)
noise = img + img * gauss

# Ensure the noisy image values are within valid range
noise = np.clip(noise, 0, 255).astype(np.uint8)

# Display images
plt.figure(num='Failure NOISE')
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(noise), plt.title('Noise')
plt.xticks([]), plt.yticks([])
plt.show()
