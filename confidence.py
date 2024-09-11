import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import os
import cv2
import torch
import json

def save_confidence_scores_to_json(file_name, models, categories, confidence_values):
    data = {}
    for i, model in enumerate(models):
        data[str(model)] = dict(zip(categories, confidence_values[i]))
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)



def get_confidence_score(model, image_folder, num):
    # List all test in the image folder
    image_folder = f"{image_folder[:-11]}/images/test/"
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    total_confidence = 0
    total_predictions = 0
    z=0
    # Loop over each image
    for image_file in image_files:
        if z==num:
            break
        image_path = os.path.join(image_folder, image_file)

        # Load the image using OpenCV
        img = cv2.imread(image_path)

        # Perform object detection using the model
        results = model(img)

        # Extract the confidence scores from the results
        for result in results:
            for box in result.boxes:
                confidence = box.conf  # Assuming `conf` is the confidence score for the bounding box
                if isinstance(confidence, torch.Tensor):  # Check if confidence is a tensor
                    confidence = confidence.item()  # Convert tensor to scalar
                total_confidence += confidence
                total_predictions += 1
        z+=1

    # Calculate the average confidence score
    if total_predictions > 0:
        average_confidence = total_confidence / total_predictions
    else:
        average_confidence = 0.0

    return average_confidence

# Load YOLO models
models = [
    YOLO('YOLOv8-large-daytime/weights/best.pt'),
    YOLO('YOLOv8-large-daytime-cond&ice/weights/best.pt'),
    YOLO('YOLOv8-large-daytime-gaussian-v2/weights/best.pt'),
    YOLO('YOLOv8-large-daytime-cond&ice-gaussian/weights/best.pt'),
    YOLO('YOLOv8-large-daytime-blur/weights/best.pt'),
    YOLO('YOLOv8-large-daytime-brightness-v2/weights/best.pt'),
    YOLO('YOLOv8-large-daytime-rain/weights/best.pt'),
    YOLO('YOLOv8-large-night-original/weights/best.pt'),
    YOLO('YOLOv8-large-night-cond&ice/weights/best.pt'),
    YOLO('YOLOv8-large-night-gaussian/weights/best.pt'),
    YOLO('YOLOv8-large-night-brightness/weights/best.pt'),
    YOLO('YOLOv8-large-night-blur/weights/best.pt'),
    YOLO('YOLOv8-large-night-rain/weights/best.pt')
]


categories = [
    'day-original', 'day-gaussian', 'day-brightness', 'day-blur', 'day-cond&ice',
    'day-rain', 'night-original', 'night-gaussian', 'night-brightness',
    'night-blur', 'night-cond&ice', 'night-rain', 'night-daytime'
]


image_folders = [
    'BDD100k-daytime/test/test/', 'BDD100k-daytime-gaussian/test/test/',
    'BDD100k-daytime-brightness/test/test/', 'BDD100k-daytime-blur/test/test/',
    'BDD100k-daytime-cond&ice/test/test/', 'BDD100k-daytime-rain/test/test/',
    'BDD100k-night/test/test/', 'BDD100k-night-gaussian/test/test/' , 'BDD100k-night-brightness/test/test/', 'BDD100k-night-blurness/test/test/' ,
    'BDD100k-night-cond&ice/test/test/', 'BDD100k-night-rain/test/test/', 'BDD100k-night-daytime/test/test/'

]

# Create a mapping from categories to image folders
category_to_image_folders = dict(zip(categories, image_folders))

# Calculate and save confidence scores for each category
for category in categories:
    # Initialize a dictionary to store confidence scores for the current category
    category_scores = {}
    if os.path.isfile(f'{category}_confidence_scores.json'):
        print("File exists. No need to do computation for this category")
        continue
    for j, model in enumerate(models):

        # Calculate the confidence score for the model on the test in the current category's folder
        confidence_score = get_confidence_score(model=model, image_folder=category_to_image_folders[category],num=20)
        print(confidence_score)
        # Store the confidence score in the dictionary
        category_scores[f'model_{j + 1}'] = confidence_score

    # Save the confidence scores for this category to a JSON file

    with open(f'{category}_confidence_scores.json', 'w') as json_file:
        json.dump(category_scores, json_file, indent=4)



