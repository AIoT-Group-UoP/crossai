from scipy import interpolate
import numpy as np
from itertools import groupby


def interpolate_preds(predictions):
    """Interpolates the predictions of a model in regard to time.

    Args:
        predictions (list): List of predictions

    Returns:
        interpolated_preds (list): List of interpolated predictions
    """

    predictions = np.array(predictions)
    predictions = np.transpose(predictions)
    interpolated_preds = []

    for i in range(len(predictions)):
        interpolated = interpolate.interp1d(np.arange(len(predictions[i])),
                                            predictions[i], kind="cubic")
        xnew = np.linspace(0, len(predictions[i]) - 1, 100)

        final = interpolated(xnew)
        interpolated_preds.append(final)

    return interpolated_preds


def count_events(predictions, pred_thres: float, consequent_frames: int = 1):
    """Counts the number of detected events (Groups of predictions above a
    threshold).

    Args:
        predictions (list): List of predictions
        pred_thres (float): Prediction threshold to be used for detection
        consequent_frames (int): Minimum number of consequent frames to be
            considered as an event.

    Returns:
        event_count (list): List of event counts per class
    """

    predictions = np.array(predictions)
    predictions = np.transpose(predictions)
    event_count = []

    for _class in predictions:
        _class[_class < pred_thres] = 0
        _class[_class >= pred_thres] = 1
        _class = [list(g) for k, g in groupby(_class) if k == 1]
        _class = [g for g in _class if len(g) >= consequent_frames]
        event_count.append(len(_class))

    return event_count
