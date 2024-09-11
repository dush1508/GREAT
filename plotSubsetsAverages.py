import matplotlib.pyplot as plt
import json
import glob
import os
from collections import namedtuple

Pair = namedtuple('Pair', ['first', 'second'])


# Function to load data from a JSON file
def load_data(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {filename}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None


# Directory containing the folders with JSON files
base_directory = 'json_files'  # Adjust this path as needed

# Define the predetermined scenario order for each set
predetermined_orders = {
    'Set1': [
        'BDD100k-daytime_1',
        'BDD100k-daytime-brightness_2',
        'BDD100k-daytime_3',
        'BDD100k-night-rain_1',
        'BDD100k-night-cond&ice_2',
        'BDD100k-night_3'
    ],
    'Set2': [
        'BDD100k-daytime_1',
        'BDD100k-daytime-brightness_2',
        'BDD100k-daytime-rain_3',
        'BDD100k-night-cond&ice_1',
        'BDD100k-night-rain_2',
        'BDD100k-night_3'
    ],
    'Set3': [
        'BDD100k-daytime-blur_1',
        'BDD100k-daytime_2',
        'BDD100k-daytime-rain_3',
        'BDD100k-night-brightness_1',
        'BDD100k-night-cond&ice_2',
        'BDD100k-night-rain_3'
    ]
    # Add more sets and their respective orders here
}

# Data structures for storing the data
data_values = {}
x_labels = {}


# Function to clean model names
def clean_model_name(name, subfolder):
    if name is None:
        return ''
    if subfolder in ['5%Confidence', 'mAP']:
        return name.replace('/weights/best.pt', '')
    return name


# Loop through each set (e.g., Set1)
for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)

    if os.path.isdir(folder_path) and folder in predetermined_orders:
        # Initialize data structures for this set
        data_values[folder] = {'5%Confidence': {}, 'mAP': {}, 'Original': {}}
        x_labels[folder] = predetermined_orders[folder]

        # Loop through subfolders inside the set (e.g., 5%Confidence, mAP, Original)
        for subfolder in ['5%Confidence', 'mAP', 'Original']:
            subfolder_path = os.path.join(folder_path, subfolder)

            if os.path.isdir(subfolder_path):
                # List of JSON files to be processed within this subfolder
                json_files = glob.glob(os.path.join(subfolder_path, '*.json'))

                for file in json_files:
                    data = load_data(file)
                    if data is None:
                        continue  # Skip files that could not be loaded

                    file_label = os.path.basename(file).split('.')[0]

                    # Extract the "final_map" values and best_model names from the JSON file
                    for scenario in x_labels[folder]:
                        if scenario in data:
                            info = data[scenario]
                            if "final_map" in info and "best_model" in info:
                                data_values[folder][subfolder][scenario] = Pair(info["final_map"], info["best_model"])

print(data_values)

# Plotting
for folder in data_values:
    plt.figure(figsize=(12, 8))

    # Create scatter plots for each subfolder
    colors = {'5%Confidence': 'blue', 'mAP': 'green', 'Original': 'red'}
    markers = {'5%Confidence': 'o', 'mAP': 's', 'Original': 'D'}

    for subfolder, color in colors.items():
        x = [f"{label[:-2]} ({i + 1})" for i, label in enumerate(x_labels[folder])]  # Omit last 2 characters and add numbering
        y = [data_values[folder][subfolder].get(scenario, Pair(None, None)).first for scenario in x_labels[folder]]
        print("Debug STATEMENT")
        print(x_labels[folder], subfolder)
        labels = [clean_model_name(data_values[folder][subfolder].get(scenario, Pair(None, None)).second, subfolder) for
                  scenario in x_labels[folder]]

        # Debug output
        print(f"Folder: {folder}, Subfolder: {subfolder}")
        print("x:", x)
        print("y:", y)
        print("Labels:", labels)

        plt.plot(x, y, color=color, marker=markers[subfolder], linestyle='-', label=subfolder)  # Trace a line across points
        plt.scatter(x, y, color=color, marker=markers[subfolder])

        # Annotate each point with the model name
        for i in range(len(x)):
            if y[i] is not None:  # Avoid annotating None values
                plt.annotate(labels[i], (x[i], y[i]), fontsize=7, ha='right')

    # Customize plot
    plt.xlabel('Scenarios')
    plt.ylabel('Final mAP Values')
    plt.legend(title='Methods')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{folder}.png", dpi=1200)

    # Show plo
