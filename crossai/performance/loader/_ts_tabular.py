import pandas as pd
from crossai.performance import pilot_label_processing
from crossai.pipelines.tabular import Tabular


def csv_loader(filename, classes: list, delimiter=',', header=0):
    """Loads an instance of tabular (csv) file and returns the
    data in the equivalent CrossAI object for pilot evaluation

    Args:
        filename (str): path to the file
        classes (list): list of class names shaped [class1, class2, ...]
        delimiter (str, optional): delimiter of the csv files. Defaults to ','.
        header (int, optional): row of the header. Defaults to 0.

    Returns:
        CrossAI object: data in the equivalent CrossAI object.
    """
    df = pd.DataFrame(columns=['instance', 'label', 'feature', 'data'])
    local_df = pd.read_csv(filename, delimiter=delimiter, header=header)
    labels, _ = pilot_label_processing(filename.replace('.csv', '.json'),
                                       classes,
                                       len(local_df))

    for i in range(len(local_df.columns)):
        df = pd.concat([df, pd.DataFrame([[0,
                                           labels,
                                           local_df.columns[i],
                                           local_df.iloc[:, i].values]],
                                         columns=['instance',
                                                  'label',
                                                  'feature',
                                                  'data'])],
                       ignore_index=True)
    crossai_object = Tabular(df)

    return crossai_object
