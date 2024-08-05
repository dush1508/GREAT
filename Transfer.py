import json
import os
import shutil


def copy_night_images(night_images_file, source_folder, destination_folder):
    # Read the night_images.json file
    with open(night_images_file, 'r') as f:
        night_images = json.load(f)

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate over each image in night_images.json
    for image in night_images:
        image_name = image.get('name')
        if image_name:
            source_path = os.path.join(source_folder, image_name)
            destination_path = os.path.join(destination_folder, image_name)

            # Check if the source image exists
            if os.path.exists(source_path):
                # Copy the image to the destination folder
                shutil.copy(source_path, destination_path)
                print(f"Copied {image_name} to {destination_folder}")
            else:
                print(f"Image {image_name} not found in {source_folder}")


# Define paths
night_images_file = 'day_images.json'
source_folder = '/Users/dush/Downloads/bdd100k-6/images/100k/train'  # Replace with your source folder path
destination_folder = '/Users/dush/Documents/day_images'  # Replace with your destination folder path

# Run the function
copy_night_images(night_images_file, source_folder, destination_folder)
