import time
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# Directory containing the images
image_dir = "KITTY/Clean/images"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# YOLO model versions
models = {
    "Nano": "yolov8n.pt",
    "Medium": "yolov8m.pt",
    "Large": "yolov8l.pt"
}

# Dictionary to store average inference times
average_times = {}

# Perform inference with each model and calculate average time
for model_name, model_path in models.items():
    # Load the YOLOv8 model
    model = YOLO(model_path)

    total_time = 0.0
    num_images = len(image_files)

    # Iterate over the images and perform inference
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        # Start timing
        start_time = time.time()

        # Perform inference
        results = model(image_path)

        # End timing
        end_time = time.time()

        # Accumulate inference time
        total_time += (end_time - start_time)

    # Calculate the average inference time
    if num_images > 0:
        average_time = total_time / num_images
        average_times[model_name] = average_time
        print(f"Average Inference Time for {model_name}: {average_time:.4f} seconds")
    else:
        print(f"No images found for {model_name}.")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.bar(average_times.keys(), average_times.values(), color=['blue', 'orange', 'green'])
plt.xlabel('YOLOv8 Model')
plt.ylabel('Average Inference Time (seconds)')
plt.title('Comparison of Average Inference Time Across YOLOv8 Models')

# Save the graph as an image file
graph_filename = "average_inference_times.png"
plt.savefig(graph_filename)
print(f"Graph saved as {graph_filename}")

# Save the average times to a text file
average_times_filename = "average_inference_times.txt"
with open(average_times_filename, 'w') as f:
    for model_name, average_time in average_times.items():
        f.write(f"{model_name}: {average_time:.4f} seconds\n")
print(f"Average inference times saved as {average_times_filename}")

plt.show()
