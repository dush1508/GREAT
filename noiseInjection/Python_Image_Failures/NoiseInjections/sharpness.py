from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

# Open the image
img = Image.open("Python_Image_Failures/sim1.jpg")

# Create a sharpness enhancer
enhancer = ImageEnhance.Sharpness(img)

# Enhance the sharpness with different factors
factor_original = 1
img_original = enhancer.enhance(factor_original)

factor_blurred = 4.0
img_blurred = enhancer.enhance(factor_blurred)

# Plot the original and adjusted images
plt.figure(num='Sharpness Adjustment')
plt.subplot(121), plt.imshow(img_original), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_blurred), plt.title('Sharpened')
plt.xticks([]), plt.yticks([])
plt.show()