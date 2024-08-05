import json
import os
from PIL import Image


def create_coco_json(night_images_file, image_folder, output_file):
    # Define class index mapping
    class_mapping = {
        "pedestrian": 1,
        "rider": 2,
        "car": 3,
        "truck": 4,
        "bus": 5,
        "train": 6,
        "motorcycle": 7,
        "bicycle": 8,
        "traffic light": 9,
        "traffic sign": 10
    }

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create category list
    for category, category_id in class_mapping.items():
        coco_format["categories"].append({
            "id": category_id,
            "name": category,
            "supercategory": "none"
        })

    # Read the night_images.json file
    with open(night_images_file, 'r') as f:
        night_images = json.load(f)

    annotation_id = 1

    # Iterate over each image in night_images.json
    for image_id, image in enumerate(night_images):
        image_name = image.get('name')
        if image_name:
            image_path = os.path.join(image_folder, image_name)

            # Check if the image exists in the specified folder
            if not os.path.exists(image_path):
                print(f"Image {image_name} not found in {image_folder}")
                continue

            # Open the image to get its dimensions
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            coco_format["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": image_width,
                "height": image_height
            })

            for label in image.get('labels', []):
                class_name = label.get('category')
                class_index = class_mapping.get(class_name, -1)
                if class_index == -1:
                    continue

                box2d = label.get('box2d', {})
                x1, y1 = box2d.get('x1'), box2d.get('y1')
                x2, y2 = box2d.get('x2'), box2d.get('y2')

                # Normalize coordinates
                x1, y1 = x1 / image_width, y1 / image_height
                x2, y2 = x2 / image_width, y2 / image_height

                width = x2 - x1
                height = y2 - y1

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index,
                    "bbox": [x1 * image_width, y1 * image_height, width * image_width, height * image_height],
                    "area": width * height * image_width * image_height,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save the COCO format JSON
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"COCO format JSON saved to {output_file}")


# Define paths
night_images_file = 'day_images.json'
image_folder = '/Users/dush/Documents/day_images'  # Replace with your source images folder path
output_file = 'day_annotations.json'  # Replace with your output file path

# Run the function
create_coco_json(night_images_file, image_folder, output_file)
