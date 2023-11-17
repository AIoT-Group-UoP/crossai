import json
import numpy as np


def pilot_label_processing(json_path: str,
                           classes: list = None,
                           n_instances: int = None):
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

    Args:
        json_path (str): Path to the json file.
        classes (list): List of classes that map to the model output.
            The given list maps to numbers incrementally.
            List has shape: ['sneeze', 'cough', ...]
        n_instances (int): Number of instances in the data. The number will
            determine the length of the label array.
    Returns:
        labels (list): list of processed labels.
        segments(list): List of label segments ([start, end, label], [...]).
    """

    if classes is None or n_instances is None:
        raise ValueError('classes and n_instances must be defined.')

    segments = []

    # load json
    with open(json_path) as f:
        data = json.load(f)
    # create label array with None class
    labels = [np.nan] * n_instances

    for label in data:
        # Get segments
        label["label"] = classes.index(label["label"])
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
