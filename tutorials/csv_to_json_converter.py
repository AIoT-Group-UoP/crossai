import os
import json
import re


def extract_csv_info(filename):
    # Define the regex pattern for _class_start_end_
    pattern = r'_(\d{2}[a-d])_(\d+)_(\d+)_'

    # Use re.search to find the pattern anywhere in the string
    match = re.search(pattern, filename)

    if match:
        # Extract matched groups
        label = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))

        # Create a dictionary with the extracted information
        data = {
            "label": label,
            "type": "samples",
            "start": start,
            "end": end
        }

        return data

    else:
        print(f"Pattern not found in filename: {filename}")
        return None


def convert_csv_to_json(csv_path):
    if os.path.isdir(csv_path):
        # Convert all CSV files in each class directory
        for class_dir in os.listdir(csv_path):
            class_path = os.path.join(csv_path, class_dir)
            if os.path.isdir(class_path):
                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if file.endswith(".csv"):
                            json_data = extract_csv_info(file)
                            json_filename = os.path.splitext(file)[0] + ".json"
                            json_path = os.path.join(root, json_filename)

                            with open(json_path, 'w') as json_file:
                                json.dump([json_data], json_file, indent=4)

        print("Converted files successfully.")

    elif os.path.isfile(csv_path) and csv_path.endswith(".csv"):
        # Convert a single CSV file
        json_data = extract_csv_info(os.path.basename(csv_path))
        json_filename = os.path.splitext(
            os.path.basename(csv_path)
        )[0] + ".json"
        json_path = os.path.join(os.path.dirname(csv_path), json_filename)

        with open(json_path, 'w') as json_file:
            json.dump([json_data], json_file, indent=4)

        print("Converted file instance successfully.")
    else:
        print("Invalid input. Provide a valid directory or CSV file path.")

