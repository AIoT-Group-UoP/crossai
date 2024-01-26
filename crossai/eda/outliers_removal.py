from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats


def _remove_outliers_z_score(
    df: pd.DataFrame,
    threshold: int = 3
) -> pd.DataFrame:
    """Removes outliers from a DataFrame based on Z-scores.

    This function calculates the Z-scores of each column and of each value in
    the DataFrame and filters out rows where any value among the
    features/columns has a Z-score greater than the specified threshold.
    The Z-score represents the number of standard deviations a value is from
    the mean.

    Args:
        df: A pandas DataFrame from which to remove outliers.
        threshold: The Z-score threshold to identify outliers. Defaults to 3.

    Returns:
        A DataFrame with outliers removed based on the Z-score method.
    """
    z_scores = np.abs(stats.zscore(df))
    indices_to_keep = (z_scores < threshold).all(axis=1)
    return df[indices_to_keep]


def _remove_outliers_iqr(
    df: pd.DataFrame,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75
) -> pd.DataFrame:
    """Removes outliers from a DataFrame based on the Interquartile Range (IQR)
        method.

    This function calculates the IQR for each column in the DataFrame and
    filters out rows where any value is outside the range defined by 1.5 times
    the IQR below the first quartile or above the third quartile. The IQR is
    the range between the first (25th percentile) and third (75th percentile)
    quartiles.

    Args:
        df: A pandas DataFrame from which to remove outliers.
        lower_quantile: The lower quantile to use for calculating the IQR.
            Defaults to 0.25.
        upper_quantile: The upper quantile to use for calculating the IQR.
            Defaults to 0.75.

    Returns:
        A DataFrame with outliers removed based on the IQR method.
    """
    Q1 = df.quantile(lower_quantile)
    Q3 = df.quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_limit = (Q1 - 1.5 * IQR)
    upper_limit = (Q3 + 1.5 * IQR)
    condition = ((df >= lower_limit) & (df <= upper_limit)).all(axis=1)
    return df[condition]


def filter_outliers(
    df: pd.DataFrame,
    target_column: str,
    outlier_method: str = 'z_score',
    threshold: float = 3,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filters out outliers from a DataFrame, class by class and feature by
        feature, using the specified outlier removal method.

    This function first removes specified columns, then iterates over each
    class in the target column, applying either the Z-score or IQR method to
    filter out outliers for each class. It's designed to handle datasets where
    outlier removal needs to be class-specific.

    Args:
        df: A pandas DataFrame containing the data.
        target_column: The name of the column in df that contains the class
            labels.
        outlier_method: The method for outlier detection and removal. Options
            are 'z_score' or 'iqr'. Defaults to 'z_score'.
        threshold: The Z-score threshold to use when outlier_method is
            'z_score'. Defaults to 3.
        lower_quantile: The lower quantile to use when outlier_method is
            'iqr'. Defaults to 0.25.
        upper_quantile: The upper quantile to use when outlier_method is
            'iqr'. Defaults to 0.75.
        exclude_columns: A list of column names to exclude from outlier
            detection. Defaults to None.

    Returns:
        A DataFrame with outliers removed according to the specified method.

    Raises:
        ValueError: If target_column is in exclude_columns, or if an invalid
            outlier_method is specified.
    """
    # Create a copy of the original DataFrame to avoid modifying the input
    df_copy = df.copy()

    # Remove specified columns
    if exclude_columns:
        if target_column in exclude_columns:
            raise ValueError("Cannot Drop Target value.")
        df_copy = df_copy.drop(exclude_columns, axis=1, errors='ignore')

    dfs_no_outliers = []
    # Iterate over unique values in the target column (classes)
    for target_value in df_copy[target_column].unique():
        # Filter rows based on the current target value (class)
        class_df = df_copy[df_copy[target_column] == target_value].drop(
            [target_column], axis=1
        )

        if outlier_method == 'z_score':
            filtered_df = _remove_outliers_z_score(class_df, threshold)
        elif outlier_method == 'iqr':
            filtered_df = _remove_outliers_iqr(class_df, lower_quantile,
                                               upper_quantile)
        else:
            raise ValueError("Invalid outlier removal method. \
                             Use 'z_score' or 'iqr'.")

        with pd.option_context("mode.copy_on_write", True):
            # append class again as column
            filtered_df[target_column] = target_value

        # append to total rows
        dfs_no_outliers.append(filtered_df)

    # Concatenate the filtered results to the overall results DataFrame
    df_dataset_no_outliers = pd.concat(dfs_no_outliers, ignore_index=True)

    # Return the DataFrame with outliers removed for all classes
    return df_dataset_no_outliers
