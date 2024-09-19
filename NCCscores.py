import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

def load_images_into_array(folder_path1, folder_path2, folder_path3,folder_path4,folder_path5,folder_path6):
    images = []
    edge_images = []  # To store edge images and their indexes
    filenames = []

    folders = [folder_path1, folder_path2, folder_path3,folder_path4,folder_path5,folder_path6]

    for i, folder_path in enumerate(folders):
        folder_images = []
        folder_filenames = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

                if img is not None:
                    folder_images.append(img)
                    folder_filenames.append(filename)
                else:
                    print(f"Warning: Unable to load {filename}")

        images.extend(folder_images)
        filenames.extend(folder_filenames)

        if folder_images:
            if i > 0:
                last_index = len(images) - len(folder_images) - 1
                edge_images.append((last_index, images[last_index]))
            first_index = len(images) - len(folder_images)
            edge_images.append((first_index, folder_images[0]))

    image_array = np.array(images)

    for index, img in edge_images:
        print(f"Edge Image Index: {index}, Image Shape: {img.shape}")

    return image_array, filenames

def normalized_cross_correlation(image1, image2):
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)

    if image1.shape != image2.shape:
        raise ValueError("The input images must have the same dimensions")

    mean_image1 = np.mean(image1)
    mean_image2 = np.mean(image2)

    numerator = np.sum((image1 - mean_image1) * (image2 - mean_image2))
    denominator = np.sqrt(np.sum((image1 - mean_image1) ** 2) * np.sum((image2 - mean_image2) ** 2))

    return numerator / denominator if denominator != 0 else 0

def compute_ncc_consecutive_images(folder_path1, folder_path2, folder_path3, folder_path4, folder_path5, folder_path6, output_json_path):
    images, filenames = load_images_into_array(folder_path1, folder_path2, folder_path3, folder_path4, folder_path5, folder_path6)

    if len(images) < 2:
        raise ValueError("There must be at least two images in the folder to compute NCC.")

    ncc_results = []

    for i in range(len(images) - 1):
        ncc_score = normalized_cross_correlation(images[i], images[i + 1])
        ncc_result = {
            "index1": i,
            "index2": i + 1,
            "image1": filenames[i],
            "image2": filenames[i + 1],
            "ncc_score": float(ncc_score)
        }
        ncc_results.append(ncc_result)

    with open(output_json_path, 'w') as json_file:
        json.dump(ncc_results, json_file, indent=4)

    return ncc_results  # Return results for plotting

def plot_ncc_scores(ncc_results,i):
    indexes = [result['index1'] for result in ncc_results]
    scores = [result['ncc_score'] for result in ncc_results]

    plt.figure(figsize=(15, 6))

    # Adjust the distance between points by scaling the x-coordinates
    scaled_indexes = [i * 100 for i in indexes]  # Adjust the factor if needed

    # Reduce the point size and make the plot clearer
    plt.scatter(scaled_indexes, scores, color='b', alpha=0.5, label='NCC Score', s=20)
    plt.plot(scaled_indexes, scores, color='b', linestyle='-', alpha=0.3)

    plt.title('NCC Scores', fontsize=16)
    plt.xlabel('Image Index', fontsize=14)
    plt.ylabel('NCC Score', fontsize=14)

    # Limit the number of x-ticks to avoid overcrowding
    plt.xticks(scaled_indexes[::200], indexes[::200], rotation=45, fontsize=10, ha='right')

    plt.ylim(min(scores) - 0.1, max(scores) + 0.1)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)  # Add grid lines
    plt.tight_layout()  # Adjust layout for better display
    plt.savefig(f'ScatterPlot{i}.png', dpi=1200)



def plot_ncc_scores2(ncc_results,i):
    indexes = [result['index1'] for result in ncc_results]
    scores = [result['ncc_score'] for result in ncc_results]

    plt.figure(figsize=(15, 6))  # Set figure size for better distribution

    # Plot a line connecting all the points
    plt.plot(indexes, scores, color='b', alpha=0.7, label='NCC Score', linestyle='-', linewidth=1)

    plt.title('NCC Scores', fontsize=16)
    plt.xlabel('Image Index', fontsize=14)
    plt.ylabel('NCC Score', fontsize=14)

    # Limit the number of x-ticks to reduce overcrowding
    plt.xticks(indexes[::200], indexes[::200], rotation=45, fontsize=10, ha='right')

    plt.ylim(min(scores) - 0.1, max(scores) + 0.1)  # Adjust y-axis limits
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()  # Adjust layout for better display
    plt.savefig(f'LinePlot{i}.png', dpi=1200)

folders_set1 = ["BDD100k-daytime/images/val", "BDD100k-daytime-brightness/images/val", "BDD100k-daytime/images/val", "BDD100k-night-rain/images/val","BDD100k-night-cond&ice/images/val","BDD100k-night/images/val"]
folders_set2 = ["BDD100k-daytime/images/val", "BDD100k-daytime-brightness/images/val", "BDD100k-daytime-rain/images/val", "BDD100k-night-cond&ice/images/val","BDD100k-night-rain/images/val","BDD100k-night/images/val" ]
folders_set3 = ["BDD100k-daytime-blur/images/val", "BDD100k-daytime/images/val", "BDD100k-daytime-rain/images/val", "BDD100k-night-brightness/images/val", "BDD100k-night-cond&ice/images/val", "BDD100k-night-rain/images/val" ]

output_json_path = "ncc_results1.json"
ncc_results = compute_ncc_consecutive_images(*folders_set1, output_json_path)
plot_ncc_scores(ncc_results,1)
plot_ncc_scores2(ncc_results,1)

output_json_path = "ncc_results2.json"
ncc_results = compute_ncc_consecutive_images(*folders_set2, output_json_path)
plot_ncc_scores(ncc_results,2)
plot_ncc_scores2(ncc_results,2)

output_json_path = "ncc_results3.json"
ncc_results = compute_ncc_consecutive_images(*folders_set3, output_json_path)
plot_ncc_scores(ncc_results,3)
plot_ncc_scores2(ncc_results,3)
