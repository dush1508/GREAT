import os
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Path to the folder containing images
folder_path = "KITTY/Clean/images"


# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Check if the file is an image
        file_path = os.path.join(folder_path, filename)

        # Run the model on the image and specify the save directory
        results = model(file_path, save_txt=True)

