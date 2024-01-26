import json
import numpy as np
import os


def pilot_label_processing(json_path: str,
                           classes: list = None,
                           n_instances: int = None,
                           sampling_rate: int = None):
    """Processes the labels of pilot data to be compared to the model's output.

        The labels must be in a json array looking like this:
        [
            {
                "label": "sneeze",
                "type": "samples",
                "start": 118,
                "end": 122
            },
            {
                "label": "sneeze",
                "type": "samples",
                "start": 168,
                "end": 194
            },
            {
                "label": "cough",
                "type": "samples",
                "start": 312,
                "end": 350
            },
        ]
        or like this if type = 'time', where the start and end values are
        given in seconds:
        [
            {
                "label": "sneeze",
                "type": "time",
                "start": 0.118,
                "end": 0.122
            },
            {
                "label": "sneeze",
                "type": "time",
                "start": 0.168,
                "end": 0.194
            },
            {
                "label": "cough",
                "type": "time",
                "start": 0.312,
                "end": 0.350
            },
        ]

    Args:
        json_path (str): Path to the json file.
        classes (list): List of classes that map to the model output.
            The given list maps to numbers incrementally.
            List has shape: ['sneeze', 'cough', ...] and must contain all
            classes that are in the json file.
        n_instances (int): Number of instances in the data. The number will
            determine the length of the label array.
        sampling_rate (int): The sampling rate of the audio data.
            Used if type = 'time'.
    Returns:
        labels (list): list of processed labels.
        segments(list): List of label segments ([start, end, label], [...]).
    """

    if classes is None or n_instances is None:
        raise ValueError('classes and n_instances must be defined.')

    segments = []

    with open(json_path) as f:
        data = json.load(f)
    # create label array with None class
    labels = [np.nan] * n_instances
    if data[0]['type'] == 'time':
        if sampling_rate is None:
            raise ValueError('sampling_rate must be defined when type = time.')

    for label in data:
        # Get segments
        label["label"] = classes.index(label["label"])
        if label['type'] == 'time':
            label['start'] = int(label['start'] * sampling_rate)
            label['end'] = int(label['end'] * sampling_rate)

        segments.append([label['start'], label['end'], label['label']])
        # map labels to label array
        for i in range(label['start'], label['end']):
            labels[i] = label['label']
    # map labels to model output
    for i in range(len(labels)):
        for c in classes:
            if labels[i] == c:
                labels[i] = classes.index(c)

    return np.array(labels), segments


def threshold_predictions(predictions, threshold: float):
    """Sets the predictions of a model to 0 if they are
        below a certain threshold.

    Args:
        predictions (list): List of predictions
        threshold (float): The threshold value

    Returns:
        thresholded_preds (np.array): Array of thresholded predictions
    """
    thresholded_preds = np.array(predictions)
    thresholded_preds[thresholded_preds < threshold] = 0

    return thresholded_preds


def check_and_create_paths(path: str):
    """Checks if a path exists and creates it if not.

    Args:
        path (str): The path to check
    """
    if not os.path.exists(path):
        os.makedirs(path)
