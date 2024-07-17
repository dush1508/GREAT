from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

# Define file paths
original_image_path = "Python_Image_Failures/sim1.jpg"
overlay_image_path = "Python_Image_Failures/broken8.jpg"
output_image_path = "Python_Image_Failures/sim1_blended.jpg"

# Check if files exist
if not os.path.exists(original_image_path):
    raise FileNotFoundError(f"Original image file not found: {original_image_path}")
if not os.path.exists(overlay_image_path):
    raise FileNotFoundError(f"Overlay image file not found: {overlay_image_path}")

# Open the original image
try:
    img = Image.open(original_image_path)
except IOError as e:
    raise IOError(f"Error opening original image: {e}")

# Open the overlay image and convert it to the same mode as the original image
try:
    img2 = Image.open(overlay_image_path)
except IOError as e:
    raise IOError(f"Error opening overlay image: {e}")

# Resize the overlay image to match the original image
img2 = img2.resize(img.size)

# Blend the images
img3 = Image.blend(img, img2, 0.35)  # adjust the alpha value as needed

# Save the blended image
img3.save(output_image_path)
print(f'Blended image saved as {output_image_path}')

# Convert images to a format suitable for matplotlib
original_img = np.array(img)
blended_img = np.array(img3)

# Display the original and blended images
plt.figure(num='BROKEN LENS')
plt.subplot(121), plt.imshow(original_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blended_img), plt.title('Broken Lens')
plt.xticks([]), plt.yticks([])
plt.show()