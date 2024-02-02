import json
import numpy as np
import logging
import itertools
from collections import Counter
import pandas as pd
import os


def calculate_eval_metrics(ground_truth_segments, predictions_segments, labels, overlap_threshold_r, label_to_ignore):
    """
    This function compared the predictions with the ground truth segments and extracts if there are any insertions,
    substitutions, deletions and also decides if the class was correctly detected. The above are stored into a
    dataframe for easy manipulation.
    Args:
        overlap_threshold_r (float):
        ground_truth_segments: Segment Collection object
        predictions_segments: Segment Collection object
        labels: (list) A list with the desired classIDs.
        overall_threshold_r (float): The IOU threshold.
        label_to_ignore (str): The label to ignore during the calculation of the metrics.
    Returns:
        A dataframe with the metrics that were extracted for each class.
    """
    y_pred = list()
    y_gt = list()
    metrics_df = pd.DataFrame(
        columns=["classID", "Deletion", "Substitution", "Insertion", "Correct", "Frequency", "Counts"])
    metrics_df["classID"] = labels
    metrics_df.fillna(value=0, inplace=True)
    metrics_df.set_index("classID", inplace=True)
    predictions_array = predictions_segments
    ground_truths_array = ground_truth_segments
    for gt in ground_truths_array:
        op_times = 0
        metrics_df.loc["{}".format(gt[2]), "Frequency"] += 1
        for pred in predictions_array:
            gt_range = range(gt[0], gt[1])
            det_range = range(pred[0], pred[1])
            if pred[0] < gt[1]:
                op_range = range(max(gt_range[0], det_range[0]), min(gt_range[-1], det_range[-1]) + 1)
                if len(op_range) > 0:
                    op_times += 1
                    if len(op_range) >= (overlap_threshold_r * len(det_range)):
                        if gt[2] == pred[2]:
                            metrics_df.loc["{}".format(gt[2]), "Correct"] += 1
                            y_pred.append(pred[2])
                            y_gt.append(gt[2])
            if op_times == 0:
                metrics_df.loc["{}".format(gt[2]), "Deletion"] += 1
                y_gt.append(gt[2])
                y_pred.append(label_to_ignore)
        else:
            continue
    for pred in predictions_array:
        op_times = 0
        for gt in ground_truths_array:
            if gt[2] != label_to_ignore:
                gt_range = range(gt[0], gt[1])
                det_range = range(pred[0], pred[1])
                if gt[0] < pred[1]:
                    inter_range = range(max(gt_range[0], det_range[0]), min(gt_range[-1], det_range[-1]) + 1)
                    union_range = range(min(gt_range[0], det_range[0]), max(gt_range[-1], det_range[-1]) + 1)
                    try:
                        iou = len(inter_range) / len(union_range)
                    except ZeroDivisionError:
                        iou = 0
                    if iou > 0:
                        op_times += 1
            else:
                continue
        if op_times == 0:
            metrics_df.loc["{}".format(pred[2]), "Insertion"] += 1
            y_pred.append(pred[2])
            y_gt.append(label_to_ignore)
    metrics_df["Reliability"] = metrics_df["Correct"] / (metrics_df["Correct"] + metrics_df["Insertion"])
    metrics_df["Detection Ratio"] = metrics_df["Correct"] / (
            metrics_df["Deletion"] + metrics_df["Substitution"] + metrics_df["Correct"])
    metrics_df.fillna(value=0, inplace=True)
    if label_to_ignore is not None:
        metrics_df.drop(["{}".format(label_to_ignore)], inplace= True)
    y_pred_count = len(predictions_array)
    return metrics_df, y_pred, y_gt, y_pred_count


def calculate_rr(df):
    """
    This function calculates the Recognition Error Ratio
    Args:
        df: A dataframe with the metrics that were extracted for each class.
    Returns:
    (float) The Recognition Error Ratio
    """
    try:
        detection_ratio = df["Correct"].sum() / df["Correct"] + (
            df["Deletion"].sum() + df["Substitution"].sum() + df["Correct"].sum())
    except ZeroDivisionError:
        detection_ratio = 0
    try:
        reliability = df["Correct"].sum() / (df["Correct"].sum() + df["Insertion"].sum())
    except: 
        reliability = 0
    try:
        rr = (df["Deletion"].sum() + df["Insertion"].sum() + df["Substitution"].sum()) / \
           (df["Deletion"].sum() + df["Correct"].sum() + df["Substitution"].sum())
        rr = np.round(rr, 3)
    except ZeroDivisionError:
        rr = 1e6
    return rr, detection_ratio, reliability

    
def convert_windows_to_classes(windows_df, rw_size, op):
    """
    Converts windows to predictions segments.
    Args:
        windows_df: (df) The dataframe of the window predictions.
        classes_stats_df: (df) The dataframe of the classes statistics.
        rw_size: (int) The rolling window size
        op: (int) The overlap percentage of the windows
        min_accepted_class_len (int, optional): The minimum accepted length of a class. If not specified,
            it is considered the mean length of the corresponding detected class.
    Returns:
        predictions: (SegmentCollection object) The segment collection object with the classes predictions
    """
    non_op_step = round(rw_size - rw_size * (op / 100))
    windows_df = calculate_windows_positions(windows_df, non_op_step, rw_size)
    windows_df = windows_df[["model_confidence", "class", "wind_start", "wind_end"]]
    windows_df = windows_df.rename(columns={"class": "Class"})
    windows_df = find_consecutive_windows(windows_df)
    list_approved = []
    for i, row in windows_df.iterrows():
        if row["Class"]:
            list_approved.append(1)
        else:
            list_approved.append(0)
    windows_df["approved"] = list_approved
    windows_df = windows_df.drop(windows_df[windows_df["approved"] == 0].index)
    windows_df = windows_df.drop(windows_df[windows_df["Confindence"] < 0.6].index)
    windows_df.pop("approved")
    windows_df.reset_index(inplace=True)
    del windows_df["index"]
    predictions_list = windows_df.values.tolist()
    return predictions_list, windows_df


def calculate_windows_positions(windows_df, non_op_step, rw_size):
    """
    Args:
        windows_df:
        rw_size:
        non_op_step:
    Returns:
    """
    starts_list = []
    ends_list = []
    for i in range(0, len(windows_df)):
        window_start = i * non_op_step
        starts_list.append(window_start)
        ends_list.append(window_start + rw_size)
    windows_df.loc[:, "wind_start"] = starts_list
    windows_df.loc[:, "wind_end"] = ends_list
    return windows_df


def detection_evaluation_report(windows_df, ground_truth_segments, op_per,
                                         window_size, labels, overlap_threshold_r, label_to_ignore= None,
                                         plot_report=False):
    """Calculated the detection metrics for a pilot instance

    Args:
        windows_df (pd.Dataframe): A dataframe with the predictions for the instance.
        ground_truth_segments (np.array): An array with the ground truth segments for the instance.
        op_per (float): The overlap between the windows.
        window_size (int): The size in samples of the windows.
        labels (list): A list with the names of the labels.
        overlap_threshold_r (flow): The IOU threshold.
        plot_report (bool, optional): _description_. Defaults to False.

    Returns:
        metrics_df (pd.Dataframe): A dataframe with the detection metrics for the instance.
        report (dict): A dictionary with the detection metrics for the instance.
    """
    report = dict()
    predictions_segments, predictions_segments_df = convert_windows_to_classes(windows_df, window_size, op_per)
    metrics_df, y_pred, y_gt, y_pred_count = calculate_eval_metrics(ground_truth_segments, predictions_segments, labels,
                                                       overlap_threshold_r, label_to_ignore)
    rr, mean_dt, mean_rel = calculate_rr(metrics_df)
    metrics_df, total_counts = calculate_frequency_counts(metrics_df, ground_truth_segments, label_to_ignore)
    report["classes_metrics"] = dict()
    report["classes_metrics"] = metrics_df.set_index("classID").to_dict("index")
    report["RER"] = rr
    report["Detection_Ratio"] = mean_dt
    report["Reliability"] = mean_rel
    report["total_count"] = total_counts
    return metrics_df, report, y_pred, y_gt, predictions_segments_df


def find_consecutive_windows(windows_df):
    """Finds predictions segments

    Args:
        windows_df (_type_): _description_

    Returns:
        _type_: _description_
    """    
    windows_df["disp"] = (windows_df.Class != windows_df.Class.shift()).cumsum()
    windows_df = pd.DataFrame({"wind_start": windows_df.groupby("disp").wind_start.first(),
                               "wind_end": windows_df.groupby("disp").wind_end.last(),
                               "Class": windows_df.groupby("disp").Class.first(),
                               "Confindence": windows_df.groupby("disp").model_confidence.mean()}).reset_index(
        drop=True)
    windows_df["Length"] = windows_df["wind_end"] - windows_df["wind_start"]
    return windows_df


def export_final_dt_report(reports_list):
    """Exports a final report on all the pilot instances that are being evaluated.

    Args:
        reports_list (list of dictionaries): A list with the dictionaries of all the pilot instances.

    Returns:
        final_report: A dictionary with the metrics for all the classes and cumulative metrics.
    """
    mean_metrics_dict = {"Reliability": 0, "Detection_Ratio": 0}
    metrics_dict = {"Correct": 0, "Deletion": 0, "Insertion": 0, "Substitution": 0}
    reports_count = len(reports_list)
    final_report = dict()
    final_report["classes_final_metrics"] = dict()
    for rep in reports_list:
        for key, _class in rep["classes_metrics"].items():
            if not key in final_report["classes_final_metrics"].keys():
                final_report["classes_final_metrics"][key] = dict()
            for metric, value in _class.items():
                if not metric in final_report["classes_final_metrics"][key].keys():
                    final_report["classes_final_metrics"][key][metric] = 0
                final_report["classes_final_metrics"][key][metric] += value
                if metric in ["Correct", "Deletion", "Insertion", "Substitution"]:
                    metrics_dict[metric] += value
        for key, metric in rep.items():
            if key in ["RR", "Reliability", "Detection_Ratio", "total_count"]:
                if not key in final_report.keys():
                    final_report[key] = 0
                if key == "total_count":
                    final_report[key] += metric
    for key, _class in final_report["classes_final_metrics"].items():
        for metric, value in _class.items():
            if metric == "Detection Ratio":
                try:
                    final_report["classes_final_metrics"][key][metric] = final_report["classes_final_metrics"][key]["Correct"] / (final_report["classes_final_metrics"][
                                                        key]["Deletion"] + final_report["classes_final_metrics"][
                                        key]["Correct"] + final_report["classes_final_metrics"][key]["Substitution"])
                except ZeroDivisionError:
                    final_report["classes_final_metrics"][key][metric] = 0.0
                mean_metrics_dict["Detection_Ratio"] += final_report["classes_final_metrics"][key][metric]
            if metric == "Reliability":
                try:
                    final_report["classes_final_metrics"][key][metric] = final_report["classes_final_metrics"][key][
                                                                              "Correct"] / (final_report["classes_final_metrics"][
                                                                                key]["Correct"] + final_report["classes_final_metrics"][key]["Insertion"])
                except ZeroDivisionError:
                    final_report["classes_final_metrics"][key][metric] = 0.0
                mean_metrics_dict["Reliability"] += final_report["classes_final_metrics"][key][metric]
    final_report["Reliability"] = mean_metrics_dict["Reliability"]/len(final_report["classes_final_metrics"])
    final_report["Detection_Ratio"] = mean_metrics_dict["Detection_Ratio"] / len(
            final_report["classes_final_metrics"])
    final_report["total_correct_count"] = metrics_dict["Correct"]
    try:
        final_report["RR"] = (metrics_dict["Deletion"] + metrics_dict["Insertion"] + metrics_dict["Substitution"]) /(metrics_dict["Deletion"] + metrics_dict["Correct"] + metrics_dict["Substitution"])
    except ZeroDivisionError:
        final_report["RR"] = 1e6
    for _class, metrics in final_report["classes_final_metrics"].items():
        for metric, value in metrics.items():
            if metric == "Frequency":
                final_report["classes_final_metrics"][_class][metric] = final_report["classes_final_metrics"][_class]["Counts"] / final_report["total_count"]
    return final_report


def calculate_frequency_counts(metrics_df, ground_truth_segments, label_to_ignore= None):
    """Calculates the frequency of appearance for each class in the 
       pilot instance.
    Args:
        metrics_df (pd.Dataframe): The dataframe with the metrics of each instance.
        ground_truth_segments (np.array): The array with the segments. 

    Returns:
        metrics_df(pd.Dataframe): The metrics_df updated with the appearance count and 
                                  frequency.
        total_count(int): The count of all the classes appearing in the pilot instance.
    """
    gt_list = []
    gt_array = ground_truth_segments
    for seg in gt_array:
        if label_to_ignore is not None:
            if seg[2] != label_to_ignore:
                gt_list.append(seg)
            else:
                continue
        else:
            gt_list.append(seg[2])
    counts = dict(Counter(gt_list))
    count_val = counts.values()
    total_count = sum(count_val)
    for key, val in counts.items():
        if label_to_ignore is not None:
            if  seg[2] != label_to_ignore:
                metrics_df.loc["{}".format(key), "Counts"] = val
                metrics_df.loc["{}".format(key), "Frequency"] = val / total_count
            else:
                continue
        else:
            metrics_df.loc["{}".format(key), "Counts"] = val
            metrics_df.loc["{}".format(key), "Frequency"] = val / total_count
    metrics_df.reset_index(inplace=True)
    return metrics_df, total_count