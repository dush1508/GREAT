import json


def filter_night_images(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Filter images with "timeofday" attribute as "night"
    night_images = [image for image in data if image.get('attributes', {}).get('timeofday') == 'daytime']

    # Write the filtered images to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(night_images, f, indent=4)

    # Print the number of images with "timeofday" as "night"
    print(f"Number of images with 'timeofday' as 'daytime': {len(night_images)}")


# Define input and output file paths
input_file = 'det_train.json'
output_file = 'day_images.json'

# Run the function
filter_night_images(input_file, output_file)
