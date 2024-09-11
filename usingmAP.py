import random
from ultralytics import YOLO
import os
import cv2
import torch
import yaml
import json

def select_images(folder_path, num_images=1000):
    all_images = [os.path.join(f"{folder_path}/images/val", f) for f in os.listdir(f"{folder_path}/images/val") if f.endswith('.jpg')]
    selected_images = random.sample(all_images, num_images)
    return selected_images

def create_temp_yaml(images, original_yaml, temp_yaml_path):
    with open(original_yaml, 'r') as f:
        data = yaml.safe_load(f)
    data['train'] = images
    data['val'] = images
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data, f)


def calculate_map(model, data_yaml):
    # Perform validation
    validation_results = model.val(data=data_yaml)

    # Print validation results for debugging
    print("Validation Results:", validation_results)

    # Check if 'box' attribute is present
    if hasattr(validation_results, 'box'):
        print("Validation Results.box:", validation_results.box)
        # Check if 'map50' attribute is present
        if hasattr(validation_results.box, 'map50'):
            return validation_results.box.map50
        else:
            print("Error: 'map50' attribute not found in 'box'")
            return None
    else:
        print("Error: 'box' attribute not found in validation_results")
        return None




folder_to_mAP = {
    'BDD100k-daytime': 'YOLOv8-large-daytime-cond&ice/weights/best.pt',
    'BDD100k-daytime-brightness': 'YOLOv8-large-daytime/weights/best.pt',
    'BDD100k-daytime-blur': 'YOLOv8-large-daytime-blur/weights/best.pt',
    'BDD100k-daytime-rain': 'YOLOv8-large-daytime/weights/best.pt',
    'BDD100k-night-rain': 'YOLOv8-large-night-rain/weights/best.pt',
    'BDD100k-night-cond&ice': 'YOLOv8-large-night-cond&ice/weights/best.pt',
    'BDD100k-night': 'YOLOv8-large-night-cond&ice/weights/best.pt',
    'BDD100k-night-brightness': 'YOLOv8-large-night-brightness/weights/best.pt'
}
# Define your folders and models
folders_set1 = ["BDD100k-daytime", "BDD100k-daytime-brightness", "BDD100k-daytime"]
folders_set2 = ["BDD100k-daytime", "BDD100k-daytime-brightness", "BDD100k-daytime-rain"]
folders_set3 = ["BDD100k-daytime-blur", "BDD100k-daytime", "BDD100k-daytime-rain"]
folders_set4 = ["BDD100k-night-rain","BDD100k-night-cond&ice","BDD100k-night"]
folders_set5 = ["BDD100k-night-cond&ice","BDD100k-night-rain","BDD100k-night"]
folders_set6 = ["BDD100k-night-brightness", "BDD100k-night-cond&ice", "BDD100k-night-rain"]



def process_folders(folders, model_paths, output_json):
    map_scores = {}
    counter = 1
    for folder in folders:
        best_model = folder_to_mAP[folder]
        #selected_images = select_images(folder, num_images=1000)
        original_yaml = f"{folder}/data.yaml"
        final_map = calculate_map(YOLO(best_model), original_yaml)
        folder_with_suffix = f"{folder}_{counter}"
        counter += 1
        map_scores[folder_with_suffix] = {"best_model": best_model, "final_map": final_map}
        print(f"Folder: {folder}, Best Model: {best_model}, Final mAP: {final_map}\n")

    with open(output_json, 'w') as json_file:
        json.dump(map_scores, json_file, indent=4)



models = [
    'YOLOv8-large-daytime/weights/best.pt',
    'YOLOv8-large-daytime-cond&ice/weights/best.pt',
    'YOLOv8-large-daytime-gaussian-v2/weights/best.pt',
    'YOLOv8-large-daytime-cond&ice-gaussian/weights/best.pt',
    'YOLOv8-large-daytime-blur/weights/best.pt',
    'YOLOv8-large-daytime-brightness/weights/best.pt',
    'YOLOv8-large-night-original/weights/best.pt',
    'YOLOv8-large-night-cond&ice/weights/best.pt',
    'YOLOv8-large-night-gaussian/weights/best.pt',
    'YOLOv8-large-night-brightness/weights/best.pt',
    'YOLOv8-large-night-blur/weights/best.pt',
    'YOLOv8-large-night-rain/weights/best.pt'
]

folder_sets = [folders_set1, folders_set2, folders_set3, folders_set4, folders_set5, folders_set6]
for i, folder_set in enumerate(folder_sets):
    process_folders(folder_set, models, f"using_mAP_set_{i+1}.json")
