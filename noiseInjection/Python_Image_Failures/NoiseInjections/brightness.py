from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
import numpy as np

# Open the image
img = Image.open("Python_Image_Failures/sim1.jpg")

# Enhance brightness
enhancer = ImageEnhance.Brightness(img)

# An enhancement factor of 1.0 gives the original image
factor = 1
img1 = enhancer.enhance(factor)

# An enhancement factor of 3.5 gives a much brighter image
factor = 4
img2 = enhancer.enhance(factor)

# Convert images to a format suitable for matplotlib
original_img = np.array(img1)
bright_img = np.array(img2)

# Display images
plt.figure(num='BRIGHTNESS')
plt.subplot(121), plt.imshow(original_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(bright_img), plt.title('Bright')
plt.xticks([]), plt.yticks([])
plt.show()
