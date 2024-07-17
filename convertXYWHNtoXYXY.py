def xywhn_to_xyxyn(x, y, w, h):
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return x_min, y_min, x_max, y_max


def transform_bounding_boxes(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    transformed_lines = []
    for line in lines:  # Skip the first line
        parts = line.strip().split()
        if len(parts) == 5:  # Assuming the line has 5 parts: class_number x y w h
            class_number, x, y, w, h = map(float, parts)
            x_min, y_min, x_max, y_max = xywhn_to_xyxyn(x, y, w, h)
            transformed_lines.append(f"{int(class_number)} {x_min} {y_min} {x_max} {y_max}")

    with open(output_file_path, 'w') as file:
        for line in transformed_lines:
            file.write(line + '\n')


# Example usage:
input_file_path = 'test.txt'
output_file_path = 'transformed_bounding_boxes.txt'
transform_bounding_boxes(input_file_path, output_file_path)
