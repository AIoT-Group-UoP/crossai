from typing import Optional, List
import numpy as np
import pandas as pd


def get_high_corr_features(
    df: pd.DataFrame,
    target: Optional[str] = None,
    threshold: float = 0.75
) -> List[str]:
    """Selects features from a DataFrame that have a high correlation with each
    other, based on a specified correlation threshold. The function optionally
    focuses on a subset of the data associated with a specific target class.
    It excludes perfect self-correlations (correlation of a
                                                        feature with itself).

    Args:
        df: The DataFrame from which features are to be selected.
        target: The target class to filter the DataFrame.
            If specified, the function considers only the rows where the class
            matches the target. Defaults to None.
        threshold: The threshold for selecting highly correlated features.
            Features with a correlation higher than this threshold will be
            selected. Defaults to 0.75.

    Returns:
        A list of feature names that have a correlation higher than the
        specified threshold. These features are selected from either the entire
        DataFrame or a subset of it, depending on whether a target class
        is specified.
    """

    # Filter the DataFrame by the target class, if specified
    if target is not None:
        new_df = df[df["class"] == target]
    new_df = df.drop(columns=["class", "class_int"], errors='ignore')

    # Get the absolute value of the correlation matrix
    corr_matrix_abs = new_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix_abs.where(
        np.triu(
            np.ones(corr_matrix_abs.shape), k=1
            ).astype(bool)
        )

    # Find features with correlation greater than `low`
    high_corr_features = [column for column in upper.columns
                          if any(upper[column] >= threshold)]

    return high_corr_features
