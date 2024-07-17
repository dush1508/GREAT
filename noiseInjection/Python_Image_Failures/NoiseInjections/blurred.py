import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('Python_Image_Failures/sim1.jpg')

# Apply blur
blur = cv2.blur(img, (5, 5))  # Change values to increment/decrement blur. 0 is "no blur"

# Convert from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(num='BLURRED')
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur_rgb), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
