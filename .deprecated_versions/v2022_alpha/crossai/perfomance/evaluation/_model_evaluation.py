from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model
import numpy as np
import seaborn as sns
from typing import Union
from alibi.confidence import TrustScore
from sklearn.decomposition import PCA
from crossai.perfomance.evaluation._confidence_booster import mc_dropout_stats
import pandas as pd
from scipy import interpolate
from scipy.signal import butter
from crossai.processing.signal._custom_processing import butterworth_filtering
from ...processing.signal._utils import interpolate_matrix
from ._trust_scores_utils import reduce_dimensionality
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt


def create_cr(test_labels: np.array, prediction_labels: np.array, target_names: list):
    """_summary_

    Args:
        test_labels (np.array): _description_
        prediction_labels (np.array): _description_
        target_names (list): _description_
    """
    cr = classification_report(test_labels, prediction_labels, target_names)

    return cr 


def create_cm(test_labels: np.array, prediction_labels: np.array, target_names: list, path: str = None):
    """_summary_

    Args:
        test_labels (np.array): _description_
        prediction_labels (np.array): _description_
        target_names (list): _description_
        path (str, optional): _description_. Defaults to None.
    """
    cm = confusion_matrix(test_labels, prediction_labels, target_names)
    confusion_matrix  = sns.heatmap(cm, annot=True, fmt='g')
    fig = confusion_matrix.get_figure()
    fig.save_fig(path)


def instance_evaluation(model : Model, 
                        gt_segments : Union[list, np.ndarray],
                        eval_data : Union[list, np.ndarray],
                        mc_drp : bool = True, 
                        mc_iterations : int = 50,
                        mc_stats : list = None, 
                        labels : list = None,
                        trust_scores : TrustScore = None, 
                        decomposer : PCA = None, 
                        data_length : int = None,
                        interpolation : bool = False, 
                        cutoff : int = 7, 
                        order : int = 2, 
                        sampling_frequency: int = None, 
                        filter_type : str = "low",
                        threshold : float = 0.7,
                        op_per : int = 50, 
                        visualize_summary : bool = False,
                        path_to_save : str = None,

):
    """_summary_

    Args:
        model (Model): _description_
        gt_segments (Union[list, np.ndarray]): _description_
        eval_data (Union[list, np.ndarray]): _description_
        mc_drp (bool, optional): _description_. Defaults to False.
        mc_iterations (int, optional): _description_. Defaults to None.
        mc_dropout_stats (list, optional): _description_. Defaults to None.
        interpolation (bool, optional): _description_. Defaults to False.
        labels (list, optional): _description_. Defaults to None.
        trust_scores (TrustScore, optional): _description_. Defaults to None.
        decomposer (PCA, optional): _description_. Defaults to None.
        data_length (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    if not mc_drp:
        summary = dict()
        predictions = model.predict(eval_data)
    else:
        predictions, summary = mc_dropout_stats(model, eval_data, mc_iterations, mc_stats)
    if trust_scores is not None:
        score, closest_class = trust_scores.score(reduce_dimensionality(eval_data, decomposer), predictions)
        summary["trust_score"] = score
        summary["closest_score"] = closest_class
    if interpolation is not None:
        interp_predictions, interp_summary = interpolation_predictions(predictions, data_length, threshold=threshold, gt_segments=gt_segments, labels = labels,
                                                                       cutoff=cutoff, sampling_frequency=sampling_frequency,
                                                                       order=order, filter_type=filter_type, plot_interp_predictions= visualize_summary, path_to_save=path_to_save)
    if visualize_summary: 
        prediction_variables = {"prediction_matrix": predictions, "input_instance_length": eval_data.shape[1],
                        "rw_overlap_percent": op_per, "labels": labels}
        plot_predictions(prediction_variables, figsize=(16, 8), confidence_threshold=0,
                     title="Predictions per window", path_to_save=path_to_save+"predictions_per_window.png")
        
    return summary, predictions, interp_predictions, interp_summary


def interpolation_predictions(predictions_matrix, length, threshold : float = 0.5, labels = None,
                              gt_segments : Union[list, np.ndarray] = None,
                              cutoff : int = None, order : int = None,
                              sampling_frequency : int = None, filter_type : str = None,
                              plot_interp_predictions : bool = False, 
                              path_to_save : str = None):
    
    gt_interp_labels = create_interp_gt_labels(gt_segments, length, labels)
    
    interp_preds = interpolate_matrix(predictions_matrix, length)
    interpolation_variables = dict()
    interpolation_variables["input_instance_length"] = 1
    interpolation_variables["rw_overlap_percent"] = 0
    interpolation_variables["prediction_matrix"] = interp_preds
    interpolation_variables["labels"] = labels
    interp_preds_filtered = filter_predictions(interp_preds,
                                                  cutoff=cutoff,
                                                  fs=sampling_frequency,
                                                  order=order,
                                                  filter_type=filter_type)
    thresholded_conf_intrerp = np.where(interp_preds_filtered > threshold, interp_preds_filtered, 0.001)
    class_conf_interp = np.max(thresholded_conf_intrerp, axis=1)
    final_interp_preds = np.argmax(thresholded_conf_intrerp, axis=1)
    final_interp_preds = np.where(class_conf_interp > 0.001, final_interp_preds, 1)
    labels_that_exist = np.unique(np.concatenate((np.unique(final_interp_preds), np.unique(gt_interp_labels)), 0))
    labels = [labels[label] for label in labels_that_exist]
    if plot_interp_predictions:
            interpolation_variables = dict()
            interpolation_variables["input_instance_length"] = 1
            interpolation_variables["rw_overlap_percent"] = 0
            interpolation_variables["prediction_matrix"] = interp_preds
            interpolation_variables["labels"] = labels
            plot_interpolation_predictions(interpolation_variables, title="Interpolated predictions before filtering", path_to_save=path_to_save + "interpolated_predictions_before_filtering.png")
            interpolation_variables["prediction_matrix"] = interp_preds_filtered
            plot_interpolation_predictions(interpolation_variables, title="Interpolated predictions after filtering", path_to_save=path_to_save + "interpolated_predictions_after_filtering.png")
            interpolation_variables["prediction_matrix"] = thresholded_conf_intrerp
            plot_interpolation_predictions(interpolation_variables, title="Interpolated predictions after thresholding", path_to_save=path_to_save + "interpolated_predictions_after_thresholding.png")
            plt.close("all")
    summary_interp = dict()
    summary_interp["class"] = final_interp_preds
    summary_interp["model_confidence"] = class_conf_interp
    summary_interp = pd.DataFrame.from_dict(summary_interp)
    return final_interp_preds, summary_interp


def create_interp_gt_labels(gt_segments, length, labels = None):
    gt_interp_labels = np.empty(length)
    gt_interp_labels[:] = np.nan
    for seg in gt_segments:
        gt_interp_labels[seg[0] - 1:seg[1] -1] = int(labels.index(seg[2]))
    gt_interp_labels = np.where(np.isnan(gt_interp_labels),1,gt_interp_labels).astype(int)
    return gt_interp_labels


def plot_predictions(prediction_variables, ax=None, figsize=(16, 8), title=None,
                     confidence_threshold=0.8, path_to_save=None):
    """
    Plots the predctions for each row of the prediction matrix. As each prediction is considered to occur for a
    rolling window segment, for each segment it is plotted that part that is not overlapped (first
    `non_overlapping_step` samples. )

    Args:
        prediction_variables (dict): Parameters that are imported from the calling function and contain the parameters
            for the segmentation of the timeseries with rolling window and the prediction_matrix.
        ax (matplotlib.axe): Description of parameter `ax`.
        figsize (tupple):
        title(str, optional):
        confidence_threshold(float): Defines above which value, someone is confident for the predicted class
            and the area of these data is colored respectively.
        path_to_save (pathlib.Path):

    Returns:
        None. The predictions confidence for each row of the predictions matrix is plotted.

    """
    instance_length = prediction_variables["input_instance_length"]
    overlap_percent = prediction_variables["rw_overlap_percent"]
    predictions_matrix = prediction_variables["prediction_matrix"]
    if isinstance(predictions_matrix, pd.DataFrame):
        predictions_matrix = prediction_variables["prediction_matrix"].values
    non_overlapping_step = instance_length - np.ceil(instance_length * (overlap_percent / 1e2)).astype(np.int32)
    plot_labels = prediction_variables["labels"]
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    if predictions_matrix.shape[1] > 10:
        LABELS_COLORS = sns.color_palette("tab20", 26)
    else:
        LABELS_COLORS = sns.color_palette("tab10", 26)
    plotted_classes = np.zeros(predictions_matrix.shape[1], )
    for i in range(0, predictions_matrix.shape[0]):
        seg_start = i * non_overlapping_step
        seg_end = seg_start + non_overlapping_step
        for cl_ind in range(0, predictions_matrix.shape[1]):
            prob_class = predictions_matrix[i, cl_ind]
            label_color = LABELS_COLORS[cl_ind]
            ax.plot([seg_start, seg_end], [prob_class, prob_class], color=label_color,
                    label=plot_labels[cl_ind] if plotted_classes[cl_ind] == 0 else None)
            if plotted_classes[cl_ind] == 0:
                plotted_classes[cl_ind] = 1
        seg_confidence = np.max(predictions_matrix[i, :])
        pred_label = np.argmax(predictions_matrix[i, :])
        label_color = LABELS_COLORS[pred_label]
        if seg_confidence > confidence_threshold:
            ax.axvspan(seg_start, seg_end, facecolor=label_color, alpha=0.3)
    ax.grid("y")
    ax.legend(bbox_to_anchor=(1.05, 0.6))
    if title is not None:
        plt.title(title)
    if path_to_save is not None:
        plt.savefig(path_to_save)
        plt.close()


def plot_interpolation_predictions(interpolation_variables, ax=None, figsize=(16, 8), title=None, path_to_save=None):
    predictions_matrix = interpolation_variables["prediction_matrix"]
    if isinstance(predictions_matrix, pd.DataFrame):
        predictions_matrix = interpolation_variables["prediction_matrix"].values
    plot_labels = interpolation_variables["labels"]
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    if predictions_matrix.shape[1] > 10:
        LABELS_COLORS = sns.color_palette("tab20", 26)
    else:
        LABELS_COLORS = sns.color_palette("tab10", 26)
    plotted_classes = np.zeros(predictions_matrix.shape[1], )
    for cl_ind in range(0, predictions_matrix.shape[1]):
        prob_class = predictions_matrix[0, cl_ind]
        label_color = LABELS_COLORS[cl_ind]
        ax.plot(np.arange(0, predictions_matrix.shape[0]), predictions_matrix[:, cl_ind], color=label_color,
                label=plot_labels[cl_ind] if plotted_classes[cl_ind] == 0 else None)
        if plotted_classes[cl_ind] == 0:
            plotted_classes[cl_ind] = 1
    ax.grid("y")
    ax.legend(bbox_to_anchor=(1.05, 0.6))
    if title is not None:
        plt.title(title)
    if path_to_save is not None:
        plt.savefig(path_to_save)
        plt.close()


def filter_predictions(data, cutoff, fs=16000 , order=4, filter_type="low", method="gust", sos=None):
    """
    Lowpass butterworth filter
    Args:
        data: numpy array (vector)
        cutoff: highest pass frequency
        fs: sampling frequency
        order: filter order
        filter_type: lowpass(low) or highpass(high)
        method: Method used for applying filter to data (usage of `scipy.filtfilt`)
    Returns: filtered array
    """
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        cutoff = np.array(cutoff)
    normal_cutoff = cutoff / nyq
    if sos is None:
        sos = butter(N=order, Wn=normal_cutoff, btype=filter_type, analog=False, output="sos")
    y = sosfiltfilt(sos, data, axis=0)
    return y