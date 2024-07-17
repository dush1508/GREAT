import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Load the original image
file_path = 'python_image_failures/sim1.jpg'  # Ensure the path is correct
img = Image.open(file_path)
img1 = cv2.imread(file_path)
if img1 is None:
    raise FileNotFoundError(f"Image file '{file_path}' not found.")

# Convert to RGB for display
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Get image dimensions
h, w, _ = img1.shape
print(f'Width: {w}, Height: {h}')

# Define number of rows and columns for the grid
num_rows = 10
num_cols = 20

# Calculate grid spacing
h_spacing = h // num_rows
w_spacing = w // num_cols

# Initialize pixel counters
count_pixel = 0

# Place 200 red pixels in a grid pattern
for row in range(num_rows):
    for col in range(num_cols):
        y = row * h_spacing + h_spacing // 2
        x = col * w_spacing + w_spacing // 2
        if 0 <= x < w and 0 <= y < h:
            img1[y, x] = (0, 0, 0)  # Red color for visibility
            count_pixel += 1

# Print the number of inserted pixels for verification
print(f'Number of pixels inserted: {count_pixel}')

# Save the modified image using OpenCV
output_file_path_cv2 = 'sim1_dead_pixels_cv2.jpg'
img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_file_path_cv2, img1_bgr)
print(f'Modified image saved as {output_file_path_cv2} using OpenCV')

# Optionally, Convert Modified Image to PIL Format and Save using PIL (Double check)
output_file_path_pil = 'sim1_dead_pixels_pil.jpg'
img1_pil = Image.fromarray(img1)
img1_pil.save(output_file_path_pil)
print(f'Modified image saved as {output_file_path_pil} using PIL')

# Display the original and modified images using Matplotlib
plt.figure(num='Fallimento DEADPIXEL Configurazione 3')
plt.subplot(121), plt.imshow(img), plt.title('Originale')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img1), plt.title('DEADPIXEL - Configurazione 3')
plt.xticks([]), plt.yticks([])
plt.show()