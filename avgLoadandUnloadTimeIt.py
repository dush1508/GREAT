import ultralytics
import gc
import timeit
import statistics
import matplotlib.pyplot as plt


# Function to load a YOLOv8 model
def load_model(model_name):
    model = ultralytics.YOLO(model_name)
    return model


# Function to unload a YOLOv8 model
def unload_model(model):
    del model
    gc.collect()  # Force garbage collection to free memory


# List of YOLOv8 model names to test
model_names = [
    'yolov8n',  # Nano model
    'yolov8s',  # Small model
    'yolov8m',  # Medium model
    'yolov8l',  # Large model
    'yolov8x',  # Extra Large model
]

# Number of iterations to repeat loading/unloading
iterations = 10

# Store load/unload times for each model
load_times = {model_name: [] for model_name in model_names}
unload_times = {model_name: [] for model_name in model_names}

# Profile each model's loading and unloading process 10 times using timeit
for model_name in model_names:
    for i in range(iterations):
        # Time loading
        load_time = timeit.timeit(stmt=f'load_model("{model_name}")', globals=globals(), number=1)
        load_times[model_name].append(load_time)

        # Load the model to unload it
        model = load_model(model_name)

        # Time unloading
        unload_time = timeit.timeit(stmt='unload_model(model)', globals=globals(), number=1)
        unload_times[model_name].append(unload_time)

# Calculate average times and standard deviations for each model
avg_load_times = []
stddev_load_times = []
avg_unload_times = []
stddev_unload_times = []

for model_name in model_names:
    avg_load_time = statistics.mean(load_times[model_name])
    stddev_load_time = statistics.stdev(load_times[model_name])
    avg_unload_time = statistics.mean(unload_times[model_name])
    stddev_unload_time = statistics.stdev(unload_times[model_name])

    avg_load_times.append(avg_load_time)
    stddev_load_times.append(stddev_load_time)
    avg_unload_times.append(avg_unload_time)
    stddev_unload_times.append(stddev_unload_time)

    print(f"{model_name} Load Time: Avg = {avg_load_time:.4f} seconds, StdDev = {stddev_load_time:.4f} seconds")
    print(f"{model_name} Unload Time: Avg = {avg_unload_time:.4f} seconds, StdDev = {stddev_unload_time:.4f} seconds")

# Plot the average load and unload times with error bars for standard deviation
plt.figure(figsize=(10, 6))

# Plotting Load Times
plt.errorbar(model_names, avg_load_times, yerr=stddev_load_times, fmt='-o', label='Load Time', capsize=5)

# Plotting Unload Times
plt.errorbar(model_names, avg_unload_times, yerr=stddev_unload_times, fmt='-o', label='Unload Time', capsize=5)

plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.title('Average Load and Unload Times for YOLOv8 Models (Using timeit)')
plt.legend()
plt.grid(True)
plt.show()
