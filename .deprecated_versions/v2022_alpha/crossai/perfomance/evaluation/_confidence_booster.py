from tensorflow.keras import Model
from typing import Union
import numpy as np
import logging 

def mc_dropout_stats(model: Model, test_data: Union[list, np.ndarray],
               iterations : int = None, statistics : list = None):
    """_summary_

    Args:
        model (Model): _description_
        test_data (Union[list, np.ndarray]): _description_
        iterations (int, optional): _description_. Defaults to None.
        statistics (list, optional): _description_. Defaults to None.
    """    
    predictions_i = model.predict(test_data)
    predictions_all = np.empty(((iterations,) + predictions_i.shape))
    predictions_all[0] = predictions_i
    for i in range(1, iterations): 
        predictions_all[i] = model.predict(test_data)
    predictions = predictions_all.mean(axis=0)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    summary = {"predicted_class": predicted_class, "confidence": confidence}
    if statistics is not None and len(statistics) <= 3:  
        assert statistics and all((isinstance(i, str) and (i == "predictions_var" or i=="predictions_stds" or i=="entropy")) for i in statistics),\
        f"The statistics arguments must be a list and the only options are predictions_var and predictions_stds"
        if "predictions_var" in statistics:
            predictions_var = predictions_all.var(axis=0)
            summary["uncertainty_var"] = calc_uncertainty_best_probability(predictions, predictions_var)
        if "predictions_stds" in statistics:
            predictions_stds = predictions_all.std(axis=0)
            summary["uncertainty_std"] = calc_uncertainty_best_probability(predictions, predictions_stds)
        if "entropy" in statistics:
            summary["entropy"] = calc_entropy(predictions)
    return predictions, summary


def calc_entropy(probs):
    """
    Calculates the entropy of the input probabilities
    Args:
        probs (np.ndarray): Array with the Monte-Carlo prediction probabilities of a classifier.
        It is expected to be of shape <Monte-Carlo iterations> X <number_of_instances> X <number of classes>

    Returns:
        entrop (float)
    """
    prob = check_probs_shape(probs)
    entrop = - (np.log(prob) * prob).sum(axis=1)
    return entrop

def check_probs_shape(probs):
    if len(probs.shape) == 3:
        prob = probs.mean(axis=0)
    elif len(probs.shape) <= 2:
        prob = probs
    else:
        msg = "Invalid probabilities vector shape {}.".format(probs.shape)
        logging.error(msg)
        raise Exception(msg)
    return prob

def calc_uncertainty_best_probability(probs, stds_mat=None):
    """
    Return the standard deviation of the most probable class.
    Args:
        probs: Array with the Monte-Carlo prediction probabilities of a classifier.
        It is expected to be of shape <Monte-Carlo iterations> X <number_of_instances> X <number of classes>
        stds_mat (numpy.ndarray):
    Returns:

    """
    prob = check_probs_shape(probs)
    idx = prob.argmax(axis=1)

    if stds_mat is None:
        std = probs[:, np.arange(len(idx)), idx].std(axis=0)
    else:
        std = stds_mat[np.arange(len(idx)), idx]
    return std