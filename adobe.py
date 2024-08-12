# Attempt using Hough Transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

def read_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    XY = df.iloc[:, 2:].values
    return XY

def plot_XYs(XY_data, title, line_segments=None):
    plt.figure(figsize=(8, 8))
    plt.plot(XY_data[:, 0], XY_data[:, 1], 'bo-', linewidth=2, markersize=5)
    if line_segments is not None:
        for segment in line_segments:
            plt.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def regularize_curve_with_hough(XY_data):
    if len(XY_data) < 2:
        return XY_data, []

    try:
        accumulator, angles, dists = hough_line(XY_data)
        lines = []
        for _, angle, dist in zip(*hough_line_peaks(accumulator, angles, dists)):
            if np.abs(angle) < np.deg2rad(10) or np.abs(angle - np.pi/2) < np.deg2rad(10):
                continue

            x0 = dist * np.cos(angle)
            y0 = dist * np.sin(angle)
            dx = np.cos(angle + np.pi/2)
            dy = np.sin(angle + np.pi/2)

            x_line = np.linspace(XY_data[:, 0].min(), XY_data[:, 0].max(), 100)
            y_line = (y0 - (x0 - x_line) * dy / dx) if dx != 0 else np.linspace(XY_data[:, 1].min(), XY_data[:, 1].max(), 100)
            lines.append(np.column_stack([x_line, y_line]))

        if lines:
            regularized_XY = np.vstack(lines)
        else:

            regularized_XY = XY_data

    except np.linalg.LinAlgError:
        regularized_XY = XY_data
        lines = []

    return regularized_XY, lines

frag0_XY = read_csv('/content/frag0.csv')
frag01_sol_XY = read_csv('/content/frag01_sol.csv')

frag0_reg, frag0_lines = regularize_curve_with_hough(frag0_XY)
frag01_sol_reg, frag01_sol_lines = regularize_curve_with_hough(frag01_sol_XY)

# Plot the original and regularized shapes
plot_XYs(frag0_XY, 'Original Fragmented Shape 0')
plot_XYs(frag0_reg, 'Regularized Shape 0', frag0_lines)

plot_XYs(frag01_sol_XY, 'Original Fragmented Shape 1')
plot_XYs(frag01_sol_reg, 'Regularized Shape 1', frag01_sol_lines)

# Save the result of the regularized shape
plt.figure(figsize=(8, 8))
plt.plot(frag01_sol_reg[:, 0], frag01_sol_reg[:, 1], 'bo-', linewidth=2, markersize=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('regularized_output.png', bbox_inches='tight')
plt.close()

"""### *Attempt 2*"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

def read_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    XY = df.iloc[:, 2:].values  # Extract only the X and Y columns
    return XY

def plot_XYs(XY_data, title, line_segments=None):
    plt.figure(figsize=(8, 8))
    plt.plot(XY_data[:, 0], XY_data[:, 1], 'bo-', linewidth=2, markersize=5)
    if line_segments is not None:
        for segment in line_segments:
            plt.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def regularize_curve_with_hough(XY_data):
    if len(XY_data) < 2:
        return XY_data, []

    try:
        accumulator, angles, dists = hough_line(XY_data)

        lines = []
        for _, angle, dist in zip(*hough_line_peaks(accumulator, angles, dists)):
            # Filter out nearly vertical or horizontal lines (if not desired)
            if np.abs(angle) < np.deg2rad(10) or np.abs(angle - np.pi/2) < np.deg2rad(10):
                continue

            x0 = dist * np.cos(angle)
            y0 = dist * np.sin(angle)
            dx = np.cos(angle + np.pi/2)
            dy = np.sin(angle + np.pi/2)

            x_line = np.linspace(XY_data[:, 0].min(), XY_data[:, 0].max(), 100)
            y_line = (y0 - (x0 - x_line) * dy / dx) if dx != 0 else np.linspace(XY_data[:, 1].min(), XY_data[:, 1].max(), 100)
            lines.append(np.column_stack([x_line, y_line]))

        if lines:
            regularized_XY = np.vstack(lines)
        else:
            regularized_XY = XY_data

    except np.linalg.LinAlgError:
        regularized_XY = XY_data
        lines = []

    return regularized_XY, lines

input_XY = read_csv('/content/frag0.csv')

regularized_XY, lines = regularize_curve_with_hough(input_XY)

# Plot the original and regularized shape
plot_XYs(input_XY, 'Original Fragmented Shape')
plot_XYs(regularized_XY, 'Regularized Shape', lines)

# Save the result of the regularized shape
plt.figure(figsize=(8, 8))
plt.plot(regularized_XY[:, 0], regularized_XY[:, 1], 'bo-', linewidth=2, markersize=5)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('/content/regularized_output.png', bbox_inches='tight')
plt.close()

## --------------------------------------------------------------------------------------------------------------------------------------------------------------
## Attempt 3: For Regularization:
import numpy as np
from svgpathtools import svg2paths, Line, CubicBezier
import matplotlib.pyplot as plt

def extract_path_segments(file_path):
    svg_paths, _ = svg2paths(file_path)
    segment_collection = []
    for svg_path in svg_paths:
        for segment in svg_path:
            if isinstance(segment, (Line, CubicBezier)):
                segment_collection.append(segment)
    total_segments = len(segment_collection)
    return segment_collection, total_segments

def identify_outer_boundary(line_segment, all_segments):
    start_coord, end_coord = line_segment.start, line_segment.end
    max_extreme = max(abs(start_coord.real), abs(start_coord.imag), abs(end_coord.real), abs(end_coord.imag))
    outer_limit = max(max(abs(seg.start.real), abs(seg.start.imag), abs(seg.end.real), abs(seg.end.imag)) for seg in all_segments)
    return max_extreme > 0.9 * outer_limit

def generate_square_boundary(boundary_segments):
    coordinates = np.array([(seg.start.real, seg.start.imag) for seg in boundary_segments] +
                           [(seg.end.real, seg.end.imag) for seg in boundary_segments])
    min_x_val, min_y_val = np.min(coordinates, axis=0)
    max_x_val, max_y_val = np.max(coordinates, axis=0)
    side_length = max(max_x_val - min_x_val, max_y_val - min_y_val)
    mid_x_val, mid_y_val = (min_x_val + max_x_val) / 2, (min_y_val + max_y_val) / 2
    half_side_len = side_length / 2
    square_corners = [
        complex(mid_x_val - half_side_len, mid_y_val - half_side_len),
        complex(mid_x_val + half_side_len, mid_y_val - half_side_len),
        complex(mid_x_val + half_side_len, mid_y_val + half_side_len),
        complex(mid_x_val - half_side_len, mid_y_val + half_side_len)
    ]
    return [Line(square_corners[i], square_corners[(i + 1) % 4]) for i in range(4)]

def modify_segments(segment_list):
    boundary_segments = [seg for seg in segment_list if identify_outer_boundary(seg, segment_list)]
    inner_segments = [seg for seg in segment_list if seg not in boundary_segments]

    final_segments = generate_square_boundary(boundary_segments)
    final_segments.extend(inner_segments)

    return final_segments

def visualize_segments(finalized_segments):
    fig, ax = plt.subplots(figsize=(6, 6))
    for seg in finalized_segments:
        if isinstance(seg, Line):
            coords = np.array([seg.start, seg.end])
        elif isinstance(seg, CubicBezier):
            t_vals = np.linspace(0, 1, 100)
            coords = np.array([seg.point(t) for t in t_vals])
        ax.plot(coords.real, coords.imag, color='blue', linewidth=2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_file_path = "/content/frag0.svg"  # Change as needed

    try:
        segments, count = extract_path_segments(input_file_path)
        adjusted_segments = modify_segments(segments)

        if adjusted_segments:
            visualize_segments(adjusted_segments)
        else:
            print("No valid segments detected.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")