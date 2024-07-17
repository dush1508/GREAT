import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the original image
file_path = 'Python_Image_Failures/sim1.jpg'  # Ensure this path is correct
img = cv2.imread(file_path)

# Check if the image was loaded correctly
if img is None:
    raise FileNotFoundError(f"Image file '{file_path}' not found.")

# Create a copy of the image to modify
img1 = img.copy()

# Get the dimensions of the image
h, w, _ = img1.shape
print(f'Width: {w}, Height: {h}')

# Number of rows and columns for dead pixel grid
num_rows = 5
num_cols = 10

# Calculate spacing between dead pixels
row_spacing = h // num_rows
col_spacing = w // num_cols

# Insert "dead" pixels in a grid pattern
count_pixel = 0
block_size = 5  # Define the size of the block

for row in range(num_rows):
    for col in range(num_cols):
        y = row * row_spacing + row_spacing // 2
        x = col * col_spacing + col_spacing // 2
        # Ensure the block position is within the image bounds
        if 0 <= x < w and 0 <= y < h:
            for dy in range(-block_size // 2, block_size // 2 + 1):
                for dx in range(-block_size // 2, block_size // 2 + 1):
                    if 0 <= y + dy < h and 0 <= x + dx < w:
                        img1[y + dy, x + dx] = (0, 0, 0)  # Set the block to red for visibility
            count_pixel += 1

# Logging the number of dead pixels inserted
print(f'Number of pixels inserted: {count_pixel}')

# Save the modified image using cv2
output_file_path = 'Python_Image_Failures/sim1_dead_pixels.jpg'
cv2.imwrite(output_file_path, img1)
print(f'Modified image saved as {output_file_path}')

# Display the original and modified images using matplotlib
plt.figure(num='Dead Pixel Configuration - 50 Red Pixels')
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('Dead Pixels')
plt.xticks([]), plt.yticks([])
plt.show()