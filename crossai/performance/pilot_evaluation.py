from sklearn.metrics import jaccard_score
import numpy as np
from crossai.ai.classify import predict_y, predict_y_per_window
import json
from typing import Literal

"""
This module contains functions for pilot data evaluation.
The model performance will be evaluated based on confidence,
uncerrtainty and trustworthiness.
"""

_RETURN_TYPES = Literal["class", "probas", "std", "variance",
                        "entropy", "mean_pred", "all"]


def intersection_over_union(pred, label):
    """Calculates the intersection over union for each class.

    Args:
        pred (numpy array): prediction probabilities.
        label (numpy array): labels of the data.

    Returns:
        iou (numpy array): intersection over union for each class.
    """
    # Convert nans to -1
    label = np.nan_to_num(label, copy=False, nan=-1)
    pred = np.nan_to_num(pred, copy=False, nan=-1)

    iou = jaccard_score(label, pred, average=None)

    # Back to nans
    label = label.astype(float)
    pred = pred.astype(float)
    label[label == -1.0] = np.nan
    pred[pred == -1.0] = np.nan
    return iou


def detection_metrics(labels,
                      predicted_classes,
                      duration_thres: int = 1):
    """Calculates the number of insertions, deletions, substitutions and
        correct predictions.

        Insertion is considered when a class is predicted in a None class.
        Deletion is considered when a None class is predicted in a class
        or detection is shorter than the GT_dur_threshold.
        Substitution is considered when a class is predicted in another
        class.
        Correct is considered when a class is predicted in the same class.

    Args:
        labels (list): labels of the data.
        predicted_classes (list): predicted classes of the data.
        duration_thres (Int): duration threshold for the class to be
                                    accepted. Measured indices.

    Returns:
        insertions (int): number of insertions.
        deletions (int): number of deletions.
        substitutions (int): number of substitutions.
        correct (int): number of correct predictions.
    """

    correct = 0
    substitutions = 0
    insertions = 0
    deletions = 0
    labels = np.nan_to_num(labels, copy=False, nan=-1)
    predicted_classes = np.nan_to_num(predicted_classes, copy=False, nan=-1)

    label_array = []
    pred_class_array = []
    label_array.append(labels[0])
    pred_class_array.append(predicted_classes[0])
    for index in range(1, len(labels)+1):
        if index != len(labels) and labels[index] == labels[index-1]:
            label_array.append(labels[index])
            pred_class_array.append(predicted_classes[index])
        if index == len(labels) or labels[index] != labels[index-1]:
            temp_counter = 1
            #: count insertions
            if label_array[-1] == -1:
                for temp_idx in range(1, len(pred_class_array)+1):
                    if temp_idx != len(pred_class_array) and\
                        (pred_class_array[temp_idx] ==
                         pred_class_array[temp_idx-1]):
                        temp_counter += 1
                    else:
                        if pred_class_array[temp_idx-1] != -1 and\
                                temp_counter >= duration_thres:
                            insertions += 1
                        temp_counter = 1
            else:
                #: count substitutions
                for temp_idx in range(1, len(pred_class_array)+1):
                    if temp_idx != len(pred_class_array) and\
                        (pred_class_array[temp_idx] ==
                         pred_class_array[temp_idx-1]):
                        temp_counter += 1
                    else:
                        if pred_class_array[temp_idx-1] !=\
                            label_array[temp_idx-1] and\
                                temp_counter >= duration_thres:
                            substitutions += 1
                        elif (pred_class_array[temp_idx-1] ==
                              label_array[temp_idx-1]) and (temp_counter >=
                                                            duration_thres):
                            correct += 1
                        else:
                            deletions += 1
                        temp_counter = 1

            if index != len(labels):
                label_array = [labels[index]]
                pred_class_array = [predicted_classes[index]]

    # Back to nans
    labels = labels.astype(float)
    predicted_classes = predicted_classes.astype(float)
    labels[labels == -1.0] = np.nan
    predicted_classes[predicted_classes == -1.0] = np.nan
    return insertions, deletions, substitutions, correct


def evaluate(model,
             data,
             labels,
             per_window: bool = True,
             repeats: int = 1,
             compute: _RETURN_TYPES = 'all',
             logging: bool = False,
             duration_thres: int = 1,
             save_path: str = None):
    """Performs the evaluation of the model on the pilot data.

    Args:
        model (tensorflow model or str): model to be evaluated.
                If str, the model will be loaded from the path.
                Currently supporting pkl, h5 and keras.
        data (numpy array): Data to be evaluated.
        labels (list): Labels of the data.
        per_window (bool, optional): If True, the model will be evaluated
            per window. Defaults to True.
        repeats (int, optional): number of times the model will be evaluated.
            Defaults to 1.
        ts_scorer (function, optional): model to be used as trust scorer.
        ts_k (int, optional): number of neighbours to be used w/ trust scorer.
            Defaults to 2.
        ts_dist_type (str, optional): type of distance to be used w/ trust
            scorer. Defaults to 'point'.
        compute:
        logging:
        duration_thres (int, optional): duration threshold for the class
            to be accepted in eval_metrics. Measured in indices.
        save_path (str, optional): When defined, the results will be saved
            on the given path.
                Path must contain the name of the file to be saved.
    """
    if not per_window:
        results = predict_y(model=model, data=data, repeats=repeats,
                            compute=compute, logging=logging)
    else:
        results = predict_y_per_window(model=model, data=data,
                                       repeats=repeats, compute=compute,
                                       logging=logging)
    try:
        ins, dels, subs, cors = detection_metrics(labels,
                                                  results['class'],
                                                  duration_thres)
    except ValueError:
        print("Error in detection metrics. Nulling results.")
        ins, dels, subs, cors = 0, 0, 0, 0
    try:
        iou = intersection_over_union(results['class'], labels)
    except ValueError:
        print("Error in intersection over union. Nulling result.")
        iou = 0

    # add insertions, deletions, substitutions, correct, iou to pred_results
    results['insertions'] = ins
    results['deletions'] = dels
    results['substitutions'] = subs
    results['correct'] = cors
    results['iou'] = iou

    if save_path is not None:
        # save the results as json
        with open(save_path, 'w') as f:
            for key in results:
                if isinstance(results[key], np.ndarray):
                    results[key] = results[key].tolist()
            json.dump(results, f)

    return results
