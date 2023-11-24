import numpy as np
import json
import os
from typing import Literal
from crossai.performance.pilot_evaluation import evaluate
from crossai.performance.loader import audio_loader, csv_loader
from crossai.performance import pilot_label_processing
from crossai.visualization import plot_ts, plot_predictions
from crossai.performance.event_detection import interpolate_preds as inter_pred
from crossai.processing.filter import butterworth_filter
from crossai.performance import threshold_predictions


_EVAL_TYPES = Literal["audio", "tabular"]


def batch_evaluate(
    modality: str,
    model,
    folder,
    classes: list = None,
    per_window: bool = True,
    pipeline=None,
    repeats: int = 1,
    ts_scorer=None,
    ts_k=2,
    ts_dist_type="point",
    duration_thres: int = 1,
    save_path: str = None,
    plot_signal=None,
    plot_results: bool = False,
    scat_color_thres: float = 0.9,
    interp_filter_order: int = 2,
    interp_filter_cutoff: float = 0.1,
    pred_thres: float = None,
    **kwargs,
):
    """
    Performs batch evaluation on a folder containing pilot data.
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
        modality (str): modality of the data. Currently supporting
                        'audio' and 'csv'.
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
        ts_scorer (str or model): Trust score model to be used. Same loading
                                mechanism as model. If none, no trust score
                                will be calculated.
        ts_k (int): number of neighbors to be considered for trust score.
        ts_dist_type (str): type of distance to be used for trust score.
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
    total_res = dict()
    for metric in [
        "Num_of_Pilot",
        "Total_insertions",
        "Total_deletions",
        "Total_substitutions",
        "Total_correct",
        "insertions",
        "deletions",
        "substitutions",
        "correct",
        "RER",
        "detection_ratio",
        "reliability",
    ]:
        total_res[metric] = 0
    for metric in ["std", "variance", "entropy"]:
        total_res[metric] = []

    if (
        model is None
        or folder is None
        or classes is None
        or pipeline is None
        or save_path is None
    ):
        raise ValueError(
            "model, folder, classes, pipeline and save_path "
            "must be provided."
        )
    # load the data
    for subdir in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subdir)):
            for file in os.listdir(os.path.join(folder, subdir)):
                if not os.path.exists(
                    os.path.join(save_path, subdir, file.split(".")[0])
                ):
                    os.makedirs(os.path.join(save_path,
                                             subdir,
                                             file.split(".")[0]))

                if not file.endswith(".json"):
                    if modality == "audio":
                        sr = kwargs.get("sr", 44100)
                        eval_object = audio_loader(
                            os.path.join(folder, subdir, file), classes, sr
                        )
                    elif modality == "tabular":
                        eval_object = csv_loader(
                            os.path.join(folder, subdir, file), classes
                        )

                    if plot_signal is not None:
                        _, segments = pilot_label_processing(
                            os.path.join(
                                folder,
                                subdir,
                                file.replace(file.split(".")[-1], "json"),
                            ),
                            classes=classes,
                            n_instances=len(eval_object.data[0])
                        )

                        # check if true or list
                        if isinstance(plot_signal, bool):
                            plot_ts(
                                eval_object.data,
                                labels=np.unique(eval_object.feature),
                                title=file,
                                show=False,
                                path_to_save=os.path.join(
                                    save_path, subdir, file[:-4], "signal.png"
                                ),
                                segments=segments,
                            )
                        else:
                            plot_ts(
                                eval_object.data,
                                labels=np.unique(eval_object.feature),
                                title=file,
                                show=False,
                                path_to_save=os.path.join(
                                    save_path, subdir, file[:-4], "signal.png"
                                ),
                                segments=segments,
                                plot_features=plot_signal,
                            )

                    pipeline.transform(eval_object)
                    filename = file.replace(file.split(".")[-1], "json")

                    pilot_save_path = os.path.join(
                        save_path, subdir, filename[:-5], "results.json"
                    )

                    results = evaluate(
                        model=model,
                        data=eval_object.data,
                        labels=eval_object.labels,
                        per_window=per_window,
                        repeats=repeats,
                        ts_scorer=ts_scorer,
                        ts_k=ts_k,
                        ts_dist_type=ts_dist_type,
                        duration_thres=duration_thres,
                        save_path=pilot_save_path,
                    )
                    total_res["Num_of_Pilot"] += 1

                    mean_std = np.mean(np.array(results["std"]), axis=0)
                    total_res["std"].append(mean_std)

                    mean_var = np.mean(np.array(results["variance"]), axis=0)
                    total_res["variance"].append(mean_var)

                    mean_entr = np.mean(np.array(results["entropy"]), axis=0)
                    total_res["entropy"].append(mean_entr)

                    total_res["Total_insertions"] += results["insertions"]
                    total_res["Total_deletions"] += results["deletions"]
                    total_res["Total_substitutions"] +=\
                        results["substitutions"]
                    total_res["Total_correct"] += results["correct"]

                    if plot_results:
                        s = kwargs.get("s", 50)
                        # plot the results
                        plot_predictions(
                            results["mean_pred"],
                            title="Predictions",
                            labels=classes,
                            color_threshold=scat_color_thres,
                            path_to_save=os.path.join(
                                save_path,
                                subdir,
                                filename[:-5],
                                "predictions.png",
                            ),
                            show=False,
                            return_artifact=False,
                            s=s,
                        )
                        if "mean_pred" in results.keys():
                            interpolated = inter_pred(results["mean_pred"])
                        else:
                            interpolated = inter_pred(results["probas"])
                        # plot the interpolated results
                        plot_ts(
                            interpolated,
                            title="Inteprolated Predictions",
                            labels=classes,
                            path_to_save=os.path.join(
                                save_path,
                                subdir,
                                filename[:-5],
                                "interpolated.png",
                            ),
                            show=False,
                            return_artifact=False,
                        )
                        fltered = butterworth_filter(
                            interpolated,
                            "lp",
                            sr=None,
                            order=interp_filter_order,
                            cutoff_high=interp_filter_cutoff,
                        )
                        # plot the filtered interpolated results
                        plot_ts(
                            fltered,
                            title=file,
                            labels=classes,
                            path_to_save=os.path.join(
                                save_path,
                                subdir,
                                filename[:-5],
                                "interpolated_filt.png",
                            ),
                            show=False,
                            return_artifact=False,
                        )
                        if pred_thres is not None:
                            thresholded = threshold_predictions(fltered,
                                                                pred_thres)
                            plot_ts(
                                thresholded,
                                title=file,
                                labels=classes,
                                filter_order=interp_filter_order,
                                filter_cutoff=interp_filter_cutoff,
                                path_to_save=os.path.join(
                                    save_path,
                                    subdir,
                                    filename[:-5],
                                    "interpolated_clip.png",
                                ),
                                show=False,
                                return_artifact=False,
                            )

                else:
                    continue
        else:
            continue

    for metric in ["std", "variance", "entropy"]:
        total_res[metric] = np.array(total_res[metric])
        total_res[metric] = np.mean(total_res[metric], axis=0).tolist()

    for metric in [
        "insertions",
        "deletions",
        "substitutions",
        "correct",
    ]:
        total_res[metric] = (total_res["Total_" + metric[5:]] /
                             total_res["Num_of_Pilot"])

    total_res["RER"] = (
        total_res["deletions"]
        + total_res["substitutions"]
        + total_res["insertions"]
    ) / (
        total_res["deletions"]
        + total_res["substitutions"]
        + total_res["correct"]
    )

    total_res["detection_ratio"] = total_res["correct"] / (
        total_res["correct"]
        + total_res["substitutions"]
        + total_res["deletions"]
    )

    total_res["reliability"] = total_res["correct"] / (
        total_res["correct"] + total_res["insertions"]
    )

    with open(os.path.join(save_path, "total_results.json"), "w") as fp:
        json.dump(total_res, fp)

    return total_res
