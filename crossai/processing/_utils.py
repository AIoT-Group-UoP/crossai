from math import ceil
import numpy as np
from sklearn.preprocessing import LabelEncoder


def find_min_val(data: list) -> int:
    """
    Find the minimum value of the data.

    Args:
        data: List of data.

    Returns:
        min_len: Minimum length of the data.
    """

    min_val = min(data[0])
    for i in range(1, len(data)):
        if min(data[i]) < min_val:
            min_val = min(data[i])
    return min_val


def find_max_val(data: list) -> int:
    """
    Find the maximum value of the data.

    Args:
        data: List of data.

    Returns:
        max_len: Maximum length of the data.
    """

    max_val = max(data[0])
    for i in range(1, len(data)):
        if max(data[i]) > max_val:
            max_val = max(data[i])
    return max_val


def find_mean_val(data: list) -> float:
    """
    Find the mean value of the data.

    Args:
        data: List of data.

    Returns:
        mean_val: Mean value of the data.
    """

    mean_val = 0
    for i in range(len(data)):
        mean_val += np.mean(data[i])
    mean_val = mean_val / len(data)
    return mean_val


def find_max_len(data: list) -> int:
    """
    Find the maximum length of the data.

    Args:
        data: List of data.

    Returns:
        max_len: Maximum length of the data.
    """

    max_len = len(data[0])
    for i in range(1, len(data)):
        if len(data[i]) > max_len:
            max_len = len(data[i])
    return max_len


def find_min_len(data: list) -> int:
    """
    Find the minimum length of the data.

    Args:
        data: List of data.

    Returns:
        min_len: Minimum length of the data.
    """

    min_len = len(data[0])
    for i in range(1, len(data)):
        if len(data[i]) < min_len:
            min_len = len(data[i])
    return min_len


def find_mean_len(data: list) -> int:
    """
    Find the mean length of the data.

    Args:
        data: List of data.

    Returns:
        mean_len: Mean length of the data.
    """

    mean_len = 0
    for i in range(len(data)):
        mean_len += len(data[i])
    mean_len = ceil(mean_len / len(data))
    return mean_len


def pad_or_trim(
    data: list,
    fill_value: int = 0,
    pad_type: str = 'max'
) -> list:
    """
    Pad or trim the data to the same length.

    Given a list of data, pads or trims the data to a common length.

    Args:
        data: List of data.
        fill_value: Value to fill the data with (default=0),
                          options: 'mean', 'min', 'max'.
        pad_type: Type of padding to use (default='max'),
                        options: 'max', 'min', 'mean'.

    Returns:
        list: List of data with the same length.
    """

    match fill_value:
        case 'mean':
            fill_value = find_mean_val(data)
        case 'min':
            fill_value = find_min_val(data)
        case 'max':
            fill_value = find_max_val(data)
        case _:
            if isinstance(fill_value, str):
                raise ValueError("fill_value must be an integer or"
                                 " one of 'mean', 'min', 'max'")
            fill_value = fill_value

    match pad_type:
        case 'max':
            target_len = find_max_len(data)
        case 'min':
            target_len = find_min_len(data)
        case 'mean':
            target_len = find_mean_len(data)

    for i in range(len(data)):
        if len(data[i]) > target_len:
            data[i] = data[i][:target_len]
        elif len(data[i]) < target_len:
            data[i] = np.pad(data[i], (0, target_len-len(data[i])),
                             mode='constant', constant_values=fill_value)

    data = np.array(data.tolist())
    return data


def encode_labels(
    y_train: list,
    y_test: list
) -> tuple:
    """
    Encode the labels to integers.

    Args:
        y_train: List of training labels.
        y_test: List of testing labels.

    Returns:
        tuple: A tuple containing two lists, the encoded training labels and the encoded testing labels.
    """

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded


def convert_to_model_shape(
    data: list,
    model_cat: str = "nn"
) -> np.ndarray:
    """
    Convert the data to the shape suitable for different types of models.

    Args:
        data: List of data.
        model_cat: Category of the model, which can be either "nn" for
                         Neural Network usage transformation or "statistical"
                         for sklearn models. Default is "nn".

    Returns:
        np.ndarray: Data reshaped according to the model category.
    """

    data = np.array(data.tolist())
    if model_cat == "nn":
        data = np.expand_dims(data, axis=-1)
    elif model_cat == "statistical":
        pass  # No transformation needed for statistical models
    else:
        raise ValueError("Invalid model_cat argument. Should be either 'nn'"
                         " or 'statistical'.")
    return data
