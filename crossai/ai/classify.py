import pickle
import numpy as np
import tensorflow as tf
from typing import Literal


_RETURN_TYPES = Literal["class", "probas", "std", "variance",
                        "entropy", "mean_pred", "all"]


def predict_y(
    model,
    data,
    per_window: bool = True,
    repeats: int = 1,
    ts_scorer=None,
    ts_k=2,
    ts_dist_type="point",
    compute: _RETURN_TYPES = "class",
    logging: bool = False
):
    """
    Performs the evaluation of the model on pilot data.
    The evaluation is based on confidence, uncertainty and trustworthiness.

    Args:
        model (str or model): Model to be evaluated.
                            If str, the model will be
                            loaded from the path. Currently
                            supporting .pkl, .h5, .keras files.
        data (pandas dataframe): data to be predicted upon.
        per_window (bool): True to perform per window evaluation.
                            Useful for MC dropout.
        GT_threshold (float): confidence threshold required for the class
                                to be accepted.
        repeats (int): number of times the evaluation will be repeated.
        ts_scorer (str or model): Trust score model to be used. Same loading
                                mechanism as model. If none, no trust score
                                will be calculated.
        ts_k (int): number of neighbors to be considered for trust score.
        ts_dist_type (str): type of distance to be used for trust score.
    """

    if isinstance(model, str):  # load the model
        if model.endswith(".pkl"):
            model = pickle.load(open(model, "rb"))
        elif model.endswith(".h5") or model.endswith(".keras"):
            model = tf.keras.models.load_model(model)
        else:
            raise Exception("Model file extension not supported yet.")

    # get the probabilities
    if hasattr(model, "predict_proba"):
        if logging:
            print("using predict_proba")
        if per_window:
            # iterate through every window and get the probabilities
            for instance in data:
                instance = np.expand_dims(instance, axis=0)
                instance = instance.reshape(instance.shape[0], -1)
                loc_probs = model.predict_proba(instance)
                if repeats > 1:
                    for _ in range(repeats - 1):
                        loc_probs = np.dstack(
                            (loc_probs, model.predict_proba(instance))
                        )
                if "probabilities" not in locals():
                    probabilities = loc_probs
                else:
                    probabilities = np.vstack((probabilities, loc_probs))
        else:
            data = data.reshape(data.shape[0], -1)
            probabilities = model.predict_proba(data)
            if repeats > 1:
                for _ in range(repeats - 1):
                    probabilities = np.dstack(
                        (probabilities, model.predict_proba(data))
                    )
    else:
        if logging:
            print("using predict")
        if per_window:
            for instance in data:
                instance = np.expand_dims(instance, axis=0)
                loc_probs = model.predict(instance, verbose=0)
                if repeats > 1:
                    for _ in range(repeats - 1):
                        loc_probs = np.dstack(
                            (loc_probs, model.predict(instance, verbose=0))
                        )
                if "probabilities" not in locals():
                    probabilities = loc_probs
                else:
                    probabilities = np.vstack((probabilities, loc_probs))
        else:
            probabilities = model.predict(data)
            if repeats > 1:
                for _ in range(repeats - 1):
                    probabilities = np.dstack(
                        (probabilities, model.predict(data, verbose=0))
                    )
#        if logging:
#            print_shapes_types(data=probabilities,
#                               data_name="Predictions",
#                               show_instance=True)

    results = dict()
    if compute in ["class", "all"]:
        results["class"] = np.argmax(probabilities, axis=1)

    if compute in ["probas", "all"]:
        results["probas"] = probabilities

    if probabilities.ndim == 3:  # If we have multiple repeats
        if compute in ["mean_pred", "all"]:
            mean_pred = np.mean(probabilities, axis=2)
            results["mean_pred"] = mean_pred
        if compute in ["std", "all"]:
            STD = np.std(probabilities, axis=2)
            results["std"] = STD
        if compute in ["variance", "all"]:
            variance = np.var(probabilities, axis=2)
            results["variance"] = variance
        if compute in ["entropy", "all"]:
            entropy = -np.sum(probabilities * np.log(probabilities), axis=2)
            results["entropy"] = entropy

    if ts_scorer is not None:
        if isinstance(ts_scorer, str):
            if ts_scorer.endswith(".pkl"):
                ts_scorer = pickle.load(open(ts_scorer, "rb"))
            else:
                raise Exception("TS Model file extension not supported yet.")
        trust_score, closest_class = ts_scorer.get_trust_score(
            data, probabilities, k=ts_k, dist_type=ts_dist_type
        )
        results["trust_score"] = trust_score
        results["closest_class"] = closest_class

    return results
