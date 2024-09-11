import json
import matplotlib.pyplot as plt
import numpy as np

# Load confidence scores from JSON files
values1 = []
values2 = []
values3 = []
values4 = []
values5 = []
values6 = []
values7 = []
values8 = []
values9 = []
values10 = []
values11 = []
values12 = []

def load_confidence_scores(categories):
    confidence_values = []
    for category in categories:
        with open(f'{category}_confidence_scores.json', 'r') as file:
            data = json.load(file)
            # Extract scores and sort by model index for consistency
            scores = [data[f'model_{i+1}'] for i in range(len(data))]
            confidence_values.append(scores)
    return confidence_values

categories = [
    'day-original', 'day-gaussian', 'day-brightness', 'day-blur', 'day-cond&ice',
    'day-rain', 'night-original', 'night-gaussian', 'night-brightness',
    'night-blur', 'night-cond&ice', 'night-rain'
]

# Load the confidence values
confidence_values = load_confidence_scores(categories)
for confidence in confidence_values:
    for i, conf in enumerate(confidence):
        if i == 0:
            values1.append(conf)
        elif i == 1:
            values2.append(conf)
        elif i == 2:
            values3.append(conf)
        elif i == 3:
            values4.append(conf)
        elif i == 4:
            values5.append(conf)
        elif i == 5:
            values6.append(conf)
        elif i == 6:
            values7.append(conf)
        elif i == 7:
            values8.append(conf)
        elif i == 8:
            values9.append(conf)
        elif i == 9:
            values10.append(conf)
        elif i == 10:
            values11.append(conf)
        elif i == 11:
            values12.append(conf)

# Labels for models
labels = [
    'YOLOv8-large-finetuned-daytime-original', 'YOLOv8-large-finetuned-daytime-cond&ice',
    'YOLOv8-large-finetuned-daytime-gaussian', 'YOLOv8-large-finetuned-daytime-cond&ice-gaussian','YOLOv8-large-finetuned-daytime-blur'
    ,'YOLOv8-large-finetuned-daytime-brightness', 'YOLOv8-large-finetuned-daytime-rain', 'YOLOv8-large-finetuned-night-original' , 'YOLOv8-large-finetuned-night-cond&ice',
    'YOLOv8-large-finetuned-night-gaussian', 'YOLOv8-large-finetuned-night-brightness',
    'YOLOv8-large-finetuned-night-blur', 'YOLOv8-large-finetuned-night-rain'
]

x = np.arange(6)
width = 0.120  # Reduced bar width to fit more bars

fig, ax = plt.subplots(figsize=(20, 8))  # Increased size for better readability

colors = [
    '#E63946',  # Warm Red
    '#F1A66A',  # Warm Orange
    '#F1C40F',  # Bright Yellow
    '#F39C12',  # Orange
    '#E67E22',  # Deep Orange
    '#D49A1D',  # Warm Amber
    '#8D5524',  # Warm Magenta
    '#A9A9A9',  # Grey
    '#6C5B7B',  # Purple
    '#C06C84',  # Pink
    '#3AAFA9',  # Turquoise
    '#2A9D8F',  # Teal
    '#1D3557'   # Deep Blue
]




# Plot each model and track the highest values
bars = [
    ax.bar(x - width*3, values1[:6], width, label=labels[0], color=colors[0]),
    ax.bar(x - width*2, values2[:6], width, label=labels[1], color=colors[1]),
    ax.bar(x - width*1, values3[:6], width, label=labels[2], color=colors[2]),
    ax.bar(x, values4[:6], width, label=labels[3], color=colors[3]),
    ax.bar(x + width*1, values5[:6], width, label=labels[4], color=colors[4]),
    ax.bar(x + width*2, values6[:6], width, label=labels[5], color=colors[5]),
    ax.bar(x + width*3, values7[:6], width, label=labels[6], color=colors[6])
]

# Plot each model and annotate each bar with its value
for bar_set in bars:
    for bar in bar_set:
        bar_height = bar.get_height()

        # Annotate each bar with its height value
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar_height + 0.01, f'{bar_height:.3f}',
            ha='center', va='bottom', fontsize=7.0, color='black'
        )

# Highlight the highest value in each category with an arrow above the text
for i, category_values in enumerate(zip(values1, values2, values3, values4, values5, values6, values7)):
    if i==6:
        break
    max_value = max(category_values)
    max_index = category_values.index(max_value)

    # Get the position and height of the bar
    bar = bars[max_index][i]
    bar_height = bar.get_height()

    # Add an arrow marker above the text, and place the text closer to the bar
    ax.annotate(
        '', xy=(bar.get_x() + bar.get_width() / 2, bar_height + 0.06),
        xytext=(bar.get_x() + bar.get_width() / 2, bar_height + 0.15),
        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8)
    )

ax.set_xlabel('Scenarios', fontsize=16)
ax.set_ylabel('Avg 5%Confidence', fontsize=16)
ax.set_title('Performances between models', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(categories[:6], fontsize=11)  # Rotated for better readability

# Adjust y-axis limits
y_min = min(min(values1), min(values2), min(values3), min(values4), min(values5), min(values6), min(values7), min(values8), min(values9)) - 0
y_max = max(max(values1), max(values2), max(values3), max(values4), max(values5), max(values6), max(values7), max(values8), max(values9)) + 0.15
ax.set_ylim([y_min, y_max])
 # Adjust layout to make room for the legend
plt.savefig('confidence_scores_plot_day_times.png', dpi=1200)

# Now, create a separate figure for the legend
fig_legend = plt.figure(figsize=(10, 4))
ax_leg = fig_legend.add_subplot(111)

# Remove plot content, only keeping the legend
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, loc='center', fontsize=12)
ax_leg.axis('off')  # Turn off axis lines for legend figure

# Save the legend as a separate image
plt.savefig('confidence_scores_day_legend.png', dpi=300, bbox_inches='tight')


