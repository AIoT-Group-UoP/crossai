import numpy as np
import json
import os
from crossai.performance.pilot_evaluation import evaluate
from crossai.performance.loader import audio_loader, csv_loader
from crossai.performance import pilot_label_processing, \
    threshold_predictions, check_and_create_paths
from crossai.visualization import plot_ts, plot_predictions
from crossai.performance.event_detection import interpolate_preds as inter_pred
from crossai.processing.filter import butterworth_filter


def batch_evaluate_audio(model, folder, classes: list = None,
                         per_window: bool = True, pipeline=None,
                         repeats: int = 1, duration_thres: int = 1,
                         save_path: str = None, plot_signal=None,
                         plot_results: bool = False,
                         scat_color_thres: float = 0.9,
                         interp_filter_order: int = 2,
                         interp_filter_cutoff: float = 0.1,
                         pred_thres: float = None, **kwargs):
    """
    Performs batch evaluation on a folder containing audio pilot data.
    The folder must have the following structure:
    folder
    ├── class1
    │   ├── data1.wav
    │   ├── data1.json
    │   ├── data2.wav
    │   ├── data2.json
    │   └── ...
    ├── class2
    │   ├── data1.wav
    │   ├── data1.json
    │   ├── ...

    Args:
        model (str or model): Model to be evaluated.
                            If str, the model will be
                            loaded from the path. Currently
                            supporting .pkl, .h5, .keras files.
        folder (str): path to the folder containing the pilot data.
        classes (list): list of class names shaped [class1, class2, ...]
        is_classification (bool): True if the model is a classification model.
        per_window (bool): True to perform per window evaluation.
        pipeline (pipeline): pipeline to be used for data preprocessing.
        GT_threshold (float): confidence threshold required for the class
                                to be accepted.
        repeats (int): number of times the evaluation will be repeated.
        duration_thres (int): minimum duration of a class to be accepted.
        save_path (str): path to save the results.
        plot_signal (bool or list): If True, the signal will be plotted.
                                    If list, the list must contain the
                                    list of indices of modalities to be plotted
                                    ex. [[0, 1, 2]] will plot the first 3
                                    modalities.
        plot_results (bool): If True, the results will be plotted.
        scat_color_thres (float): Prediction threshold for the
                                instance to be colored with the class color
                                in a vertical line. Default 0.5
        interp_filter_order (int): Order of the filter to be used for
                                    interpolation. Default 2.
        interp_filter_cutoff (float): Cutoff frequency of the filter to be
                                    used for interpolation. Default 0.1.
        pred_thres (float): Threshold for the predictions. If None, no
                            thresholding will be performed.
        **kwargs: additional arguments for the respective data loaders
                or plotters (e.g sr for audio data loader
                or s for predictions scatter plotter)
    """
    total_res = init_metrics()
    sr = kwargs.get("sr", 44100)
    s = kwargs.get("s", 50)

    if (model is None or folder is None
            or classes is None or save_path is None):
        raise ValueError("model, folder, classes and save_path must be"
                         " provided.")
    # load the data
    for subdir in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, subdir)):
            check_and_create_paths(os.path.join(save_path, subdir,
                                                file.split(".")[0]))
            if not file.endswith(".json"):
                eval_object = audio_loader(
                    os.path.join(folder, subdir, file), classes, sr
                    )

                if plot_signal is not None:
                    _, segments = pilot_label_processing(
                        os.path.join(
                            folder, subdir,
                            file.replace(file.split(".")[-1], "json")),
                        classes=classes,
                        n_instances=len(eval_object.data[0]),
                        sampling_rate=sr)

                    plot_ts(eval_object.data, labels="Signal", title=file,
                            show=False,
                            path_to_save=os.path.join(save_path,
                                                      subdir, file[:-4],
                                                      "signal.png"),
                            segments=segments)

                if pipeline is not None:
                    eval_object = pipeline.fit_transform(eval_object)

                filename = file.replace(file.split(".")[-1], "json")

                pilot_save_path = os.path.join(
                    save_path, subdir, filename[:-5], "results.json")
                results = evaluate(model=model,
                                   data=np.array(eval_object.data),
                                   labels=eval_object.labels,
                                   per_window=per_window, repeats=repeats,
                                   duration_thres=duration_thres,
                                   save_path=pilot_save_path
                                   )

                total_res = append_to_total_results(total_res, results)

                if plot_results:
                    plot_wrapper(s, results, classes, scat_color_thres,
                                 interp_filter_order, interp_filter_cutoff,
                                 os.path.join(save_path, subdir,
                                              filename[:-5]),
                                 pred_thres)

    total_res = finalize_total_metrics(total_res)

    with open(os.path.join(save_path, "total_results.json"), "w") as fp:
        json.dump(total_res, fp)

    return total_res


def batch_evaluate_tabular(model, folder, classes: list = None,
                           per_window: bool = True, pipeline=None,
                           repeats: int = 1, duration_thres: int = 1,
                           save_path: str = None, plot_signal=None,
                           plot_results: bool = False,
                           scat_color_thres: float = 0.9,
                           interp_filter_order: int = 2,
                           interp_filter_cutoff: float = 0.1,
                           pred_thres: float = None, **kwargs):
    """
    Performs batch evaluation on a folder containing tabular pilot data.
    The folder must have the following structure:
    folder
    ├── class1
    │   ├── data1.wav (or .csv)
    │   ├── data1.json
    │   ├── data2.wav (or .csv)
    │   ├── data2.json
    │   └── ...
    ├── class2
    │   ├── data1.wav (or .csv)
    │   ├── data1.json
    │   ├── ...

    Args:
        model (str or model): Model to be evaluated.
                            If str, the model will be
                            loaded from the path. Currently
                            supporting .pkl, .h5, .keras files.
        folder (str): path to the folder containing the pilot data.
        classes (list): list of class names shaped [class1, class2, ...]
        is_classification (bool): True if the model is a classification model.
        per_window (bool): True to perform per window evaluation.
        pipeline (pipeline): pipeline to be used for data preprocessing.
        GT_threshold (float): confidence threshold required for the class
                                to be accepted.
        repeats (int): number of times the evaluation will be repeated.
        duration_thres (int): minimum duration of a class to be accepted.
        save_path (str): path to save the results.
        plot_signal (bool or list): If True, the signal will be plotted.
                                    If list, the list must contain the
                                    list of indices of modalities to be plotted
                                    ex. [[0, 1, 2]] will plot the first 3
                                    modalities.
        plot_results (bool): If True, the results will be plotted.
        scat_color_thres (float): Prediction threshold for the
                                instance to be colored with the class color
                                in a vertical line. Default 0.5
        interp_filter_order (int): Order of the filter to be used for
                                    interpolation. Default 2.
        interp_filter_cutoff (float): Cutoff frequency of the filter to be
                                    used for interpolation. Default 0.1.
        pred_thres (float): Threshold for the predictions. If None, no
                            thresholding will be performed.
        **kwargs: additional arguments for the respective data loaders
                or plotters (e.g sr for audio data loader
                or s for predictions scatter plotter)
    """
    total_res = init_metrics()
    s = kwargs.get("s", 50)

    if (model is None or folder is None
            or classes is None or save_path is None):
        raise ValueError("model, folder, classes and save_path must be"
                         " provided.")
    # load the data
    for subdir in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, subdir)):
            check_and_create_paths(os.path.join(save_path, subdir,
                                                file.split(".")[0]))
            if not file.endswith(".json"):
                filename = os.path.join(folder, subdir, file)
                eval_object = csv_loader(filename, classes)

                if plot_signal is not None:
                    _, segments = pilot_label_processing(
                        os.path.join(
                            folder, subdir,
                            file.replace(file.split(".")[-1], "json")),
                        classes=classes,
                        n_instances=len(eval_object.data[0]))

                    plot_ts(eval_object.data,
                            labels=np.unique(eval_object.feature), title=file,
                            show=False,
                            path_to_save=os.path.join(
                                save_path, subdir, file[:-4], "signal.png"),
                            segments=segments,
                            plot_features=plot_signal)

                if pipeline is not None:
                    pipeline.transform(eval_object)

                filename = file.replace(file.split(".")[-1], "json")

                pilot_save_path = os.path.join(
                    save_path, subdir, filename[:-5], "results.json"
                    )

                results = evaluate(model=model, data=eval_object.data,
                                   labels=eval_object.labels,
                                   per_window=per_window, repeats=repeats,
                                   duration_thres=duration_thres,
                                   save_path=pilot_save_path)

                total_res = append_to_total_results(total_res, results)

                if plot_results:
                    plot_wrapper(s, results, classes, scat_color_thres,
                                 interp_filter_order, interp_filter_cutoff,
                                 os.path.join(save_path, subdir,
                                              filename[:-5]),
                                 pred_thres)

    total_res = finalize_total_metrics(total_res)

    with open(os.path.join(save_path, "total_results.json"), "w") as fp:
        json.dump(total_res, fp)

    return total_res


def init_metrics():
    total_results = dict()
    for metric in ["Num_of_Pilot", "Total_insertions", "Total_deletions",
                   "Total_substitutions", "Total_correct", "insertions",
                   "deletions", "substitutions", "correct", "RER",
                   "detection_ratio", "reliability"]:
        total_results[metric] = 0
    for metric in ["std", "variance", "entropy"]:
        total_results[metric] = []

    return total_results


def append_to_total_results(total_res, results):
    total_res["Num_of_Pilot"] += 1
    mean_std = np.mean(np.array(results.get("std", [])), axis=0)
    total_res["std"].append(mean_std)
    mean_var = np.mean(np.array(results.get("variance", [])), axis=0)
    total_res["variance"].append(mean_var)
    mean_entr = np.mean(np.array(results.get("entropy", [])), axis=0)
    total_res["entropy"].append(mean_entr)
    total_res["Total_insertions"] += results.get("insertions", 0)
    total_res["Total_deletions"] += results.get("deletions", 0)
    total_res["Total_substitutions"] += results.get("substitutions", 0)
    total_res["Total_correct"] += results.get("correct", 0)

    return total_res


def plot_wrapper(s, results, classes, scat_color_thres, interp_filter_order,
                 interp_filter_cutoff, sav_path, pred_thres=None):

    if "mean_pred" in results.keys():
        probs = results["mean_pred"]
    else:
        probs = results["probas"]

    plot_predictions(
        probs, title="Predictions", labels=classes,
        color_threshold=scat_color_thres,
        path_to_save=os.path.join(sav_path, "predictions.png"), show=False,
        return_artifact=False, s=s,)
    try:
        interpolated = inter_pred(probs)
    except:
        return

    # plot the interpolated results
    plot_ts(interpolated, title="Inteprolated Predictions", labels=classes,
            path_to_save=os.path.join(sav_path, "interpolated.png"),
            show=False, return_artifact=False)

    filtered = butterworth_filter(interpolated, "lp", sr=None,
                                  order=interp_filter_order,
                                  cutoff_high=interp_filter_cutoff)
    # plot the filtered interpolated results
    plot_ts(filtered, title="Filtered Inteprolated Predictions",
            labels=classes,
            path_to_save=os.path.join(sav_path, "interpolated_filt.png"),
            show=False, return_artifact=False)

    if pred_thres is not None:
        thresholded = threshold_predictions(filtered, pred_thres)
        plot_ts(thresholded, title="Thresholded Predictions", labels=classes,
                filter_order=interp_filter_order,
                filter_cutoff=interp_filter_cutoff,
                path_to_save=os.path.join(sav_path, "interpolated_clip.png"),
                show=False, return_artifact=False)


def finalize_total_metrics(total_res):

    for metric in ["std", "variance", "entropy"]:
        total_res[metric] = np.array(total_res[metric])
        total_res[metric] = np.mean(total_res[metric], axis=0).tolist()

    for metric in ["insertions", "deletions", "substitutions", "correct"]:
        total_res[metric] = (total_res["Total_" + metric] /
                             total_res["Num_of_Pilot"])

    total_res["RER"] = (
        total_res["deletions"]
        + total_res["substitutions"]
        + total_res["insertions"]
    ) / (
        total_res["deletions"]
        + total_res["substitutions"]
        + total_res["correct"]
    ) if total_res["deletions"]\
        + total_res["substitutions"] + total_res["correct"] != 0 else 0

    total_res["detection_ratio"] = total_res["correct"] / (
        total_res["correct"]
        + total_res["substitutions"]
        + total_res["deletions"]
    ) if total_res["correct"]\
        + total_res["substitutions"] + total_res["deletions"] != 0 else 0

    total_res["reliability"] = total_res["correct"] / (
        total_res["correct"] + total_res["insertions"]
    ) if total_res["correct"] + total_res["insertions"] != 0 else 0

    return total_res
