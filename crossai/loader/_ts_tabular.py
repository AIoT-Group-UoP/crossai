import os
import csv
import pandas as pd
import numpy as np
from crossai.loader._utils import get_sub_dirs


def csv_loader(path, delimiter=',', header=0):
    """Loads multi-axis data from csv files. The csv files must be
    organized in subdirectories, each subdirectory containing the csv files
    of a class. The csv files must have the same headers. If not, a warning
    will be displayed and the files with different headers will be ignored.

    Args:
        path (str): Path to the directory containing the csv files
        delimiter (str, optional): Delimiter of the csv files. Defaults to ','.
        header (int, optional): Row of the header. Defaults to 0.

    Returns:
        df: pandas Dataframe containing the data.
    """

    warning_flag = 0
    instance_counter = 0
    # load the headers of the first csv file
    with open(path +
              '/' + get_sub_dirs(path)[0][0] +
              '/' + os.listdir(path + '/' +
                               get_sub_dirs(path)[0][0])[0], 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)

    df = pd.DataFrame(columns=['instance', 'label', 'feature', 'data'])

    # for each subdirectory in the path read the csv files contained in it
    for subdir in get_sub_dirs(path)[0]:
        files = os.listdir(path + '/' + subdir)
        for file in files:
            instance_counter += 1
            local_df = pd.read_csv(path + '/' + subdir + '/' + file,
                                   delimiter=delimiter,
                                   header=header)
            # check if the headers are the same
            if local_df.columns.tolist() != headers:
                if warning_flag == 0:
                    print('Warning! Different headers detected in ' +
                          path + '/' + subdir + '/' + file +
                          '. This file and every files that do not have"\
                          " the same headers as the first file (' +
                          headers +
                          ') will be ignored. This message will be'
                          ' displayed only once.')
                    warning_flag = 1
                instance_counter -= 1
                continue
            # add csv to dataframe. Each csv column goes to a different row
            # of the datafrane
            for i in range(len(local_df.columns)):
                data = local_df.iloc[:, i].values.astype(np.float32)
                df = pd.concat([df,
                                pd.DataFrame([[instance_counter,
                                               subdir,
                                               local_df.columns[i],
                                               data]],
                                             columns=['instance',
                                                      'label',
                                                      'feature',
                                                      'data'])],
                               ignore_index=True)
    print('Loaded classes: ' + str(get_sub_dirs(path)[1]))
    return df
