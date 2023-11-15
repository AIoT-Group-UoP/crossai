import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import glob
import copy
from tqdm import tqdm
from scipy.io import wavfile


# wavfile reader function needed for loading the audio data
def audio_loader(path, sr=22500, n_workers=min(mp.cpu_count(), 4)):
    """
    Loads the audio data from a directory and returns the data
    as a pandas dataframe

    The directory should :
        - contain subdirectories containing the wav files
            - the name of the subdirectories should be the label
            of the wav files
        or
        - contain wav files itself
            - the name of the directory will be the label of the wav files

    Args:
        path (str): path to the directory
        sr (int, optional): sampling rate of the audio data. Defaults to 44100.
        n_workers (int, optional): number of workers for multiprocessing.
                                    Defaults to mp.cpu_count().

    Returns:
        pandas dataframe: data from the wav files in a pandas dataframe
     """

    data = []  # sound data
    df = []  # dataframes
    subdirs, subdirnames = get_sub_dirs(path)

    global sampling_rate
    sampling_rate = copy.deepcopy(sr)

    progress = tqdm(total=len(subdirs) + 2,
                    desc="Loading audio data from {}".format(
                    subdirnames), position=0, leave=True)  # progress bar init

    # load the sound data using multiprocessing
    for i in subdirs:
        pool = mp.get_context("fork").Pool(n_workers)
        data.append(pool.map(wavfile_reader, glob.glob(
            os.path.join(path, i, '*.wav'))))
        progress.update(1)
        progress.set_description("Loaded data from {}".format(i))
        pool.close()
        pool.join()

    progress.update(1)
    progress.set_description("Load data into a dataframe")

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j].astype(np.float32), subdirnames[i])

    df = pd.DataFrame(columns=['data', 'label', 'indice'])

    for i in range(len(data)):
        for j in range(len(data[i])):
            df.loc[len(df)] = [data[i][j][0], data[i][j][1], j]

    progress.update(1)
    progress.set_description("Loaded data into the dataframe")

    return df
