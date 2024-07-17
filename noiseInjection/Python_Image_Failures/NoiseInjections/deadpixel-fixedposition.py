import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Load the original image
file_path = 'python_image_failures/sim1.jpg'  # Ensure this path is correct
img = cv2.imread(file_path)
if img is None:
    raise FileNotFoundError(f"Image file '{file_path}' not found.")

# Convert the image to RGB (from BGR, which OpenCV uses by default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a copy of the image to modify
img1 = img.copy()

# Get the dimensions of the image
h, w, _ = img1.shape

# Coordinates to set a red pixel
x = 150
y = 90

# Check if the coordinates are within the image bounds
if 0 <= x < w and 0 <= y < h:
    # Set a 3x3 square of pixels to red color for visibility
    img1[max(y-1, 0):min(y+2, h), max(x-1, 0):min(x+2, w)] = (255, 0, 0)  # Red color
else:
    print(f"Coordinates ({y}, {x}) are out of image bounds.")

# Save the modified image as a separate file
output_path = 'python_image_failures/sim1_modified.jpg'
img1_pil = Image.fromarray(img1)
img1_pil.save(output_path)
print(f"Modified image saved as '{output_path}'")

# Display the original and modified images using Matplotlib
plt.figure(num='Fallimento DEADPIXEL')
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img1), plt.title('DeadPixel')
plt.xticks([]), plt.yticks([])
plt.show()