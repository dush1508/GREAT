from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# Open the image
picture1 = Image.open("Python_Image_Failures/sim1.jpg")
picture = Image.open("Python_Image_Failures/sim1.jpg")

# Load pixels and get dimensions
pixels = picture.load()
width, height = picture.size

# Set all pixels to black
for i in range(width):
    for j in range(height):
        pixels[i,j] = (0, 0, 0)

# Convert images to a format suitable for matplotlib
original_img = np.array(picture1)
black_img = np.array(picture)

# Display images
plt.figure(num='BLACK')
plt.subplot(121), plt.imshow(original_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(black_img), plt.title('Black')
plt.xticks([]), plt.yticks([])
plt.show()
