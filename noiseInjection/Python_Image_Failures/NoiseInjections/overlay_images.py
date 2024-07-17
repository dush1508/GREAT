from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np

output_image_path = "Python_Image_Failures/sim1_blended2.jpg"

# Open the original image
img = Image.open("Python_Image_Failures/sim1.jpg").convert("RGB")
img3 = img.copy()

# Open the overlay image and ensure it's in RGBA format
img2 = Image.open("Python_Image_Failures/rain/rain3.png").convert("RGBA")

# Overlay the image
img3.paste(img2, (0, 0), img2)

# Save the blended image
img3.save(output_image_path)
print(f'Blended image saved as {output_image_path}')

# Convert images to a format suitable for matplotlib
original_img = np.array(img)
overlaid_img = np.array(img3)

# Display images
plt.figure(num='Overlay image')
plt.subplot(121), plt.imshow(original_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(overlaid_img), plt.title('Overlaid image')
plt.xticks([]), plt.yticks([])
plt.show()
