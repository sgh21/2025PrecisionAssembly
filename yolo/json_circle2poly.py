import os
import json
import numpy as np

def circ2poly(json_data):
    for shape in json_data['shapes']:
        if shape['shape_type']=='circle' and len(shape['points']) == 2:
            center = np.array(shape['points'][0])
            roundpoint  = np.array(shape['points'][1])
            radius = np.linalg.norm(roundpoint - center)
            print(f"Converting circle at {center} with radius {radius} to polygon")
            # Calculate the points of the polygon approximation
            num_points = 32  # Number of points to approximate the circle
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            polygon_points = [
                (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
                for angle in angles
            ]
            # Update the shape to be a polygon
            shape['shape_type'] = 'polygon'
            shape['points'] = polygon_points
    return json_data


if __name__ == "__main__":
    # Example JSON data structure
    path = os.path.dirname(__file__)
    path = os.path.join(path, 'dataset','origin','train','images')
    for file in os.listdir(path):
        if file.endswith('.json'):
            print(f"Processing file: {file}")
            with open(os.path.join(path, file), 'r') as f:
                json_data = json.load(f)
                data = circ2poly(json_data)
            # Save the modified json_data back to file if needed
            with open(os.path.join(path, file), 'w') as f:
                json.dump(data, f, indent=4)
                