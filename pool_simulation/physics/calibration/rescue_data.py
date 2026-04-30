import json


def rescue_data(filename, old_fps=60.0, true_fps=30.0):
    with open(filename, 'r') as f:
        data = json.load(f)

    for pt in data:
        # Re-scale the timestamps to match the true frame rate
        pt['time'] = pt['time'] * (old_fps / true_fps)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    for i in range(1, 9):
        try:
            rescue_data(f"shot{i}_data.json")
            print(f"Rescued shot{i}_data.json!")
        except FileNotFoundError:
            pass