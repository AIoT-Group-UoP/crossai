import numpy as np
from typing import Literal

_RETURN_TYPES = Literal["class", "probas", "std", "variance",
                        "entropy", "mean_pred", "all"]


def predict_y_per_window(
    model,
    data,
    repeats: int = 1,
    compute: _RETURN_TYPES = "class",
    logging: bool = False
):
    # get the probabilities
    if hasattr(model, "predict_proba"):
        if logging:
            print("using predict_proba")
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
        if logging:
            print("using predict")
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

    results = compute_results(probabilities, compute)

    return results


def predict_y(
    model,
    data,
    repeats: int = 1,
    compute: _RETURN_TYPES = "class",
    logging: bool = False
):
    """
    Performs the evaluation of the model on pilot data.
    The evaluation is based on confidence, uncertainty and trustworthiness.

    Args:
        model (model): Model to be evaluated.
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

    # get the probabilities
    if hasattr(model, "predict_proba"):
        if logging:
            print("using predict_proba")
        probabilities = model.predict_proba(data)
        if repeats > 1:
            for _ in range(repeats - 1):
                probabilities = np.dstack(
                    (probabilities, model.predict_proba(data))
                )
    else:
        if logging:
            print("using predict")

        probabilities = model.predict(data, verbose=0)
        if repeats > 1:
            for _ in range(repeats - 1):
                probabilities = np.dstack(
                    (probabilities, model.predict(data, verbose=0))
                )

    results = compute_results(probabilities, compute)

    return results


def compute_results(probabilities, compute: _RETURN_TYPES = "class"):
    results = dict()
    if compute in ["class", "all"]:
        if probabilities.ndim == 3:
            results["class"] = np.argmax(np.mean(probabilities, axis=2),
                                         axis=1)
        else:
            try:
                results["class"] = np.argmax(probabilities, axis=1)
            except:
                results["class"] = np.argmax(probabilities)

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

    return results
