import os
import glob
import copy
import multiprocessing as mp
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from crossai.processing import resample_sig
from crossai.loader._utils import get_sub_dirs


def wavfile_reader(filename):
    """Reads a wav file and returns the data as numpy array.

    Args:
        filename (str): Path to the wav file.

    Returns:
        numpy array: Data from the wav file.
    """

    sr, signal = wavfile.read(filename)
    signal = signal.astype(np.float32)

    #convert to mono if the signal is stereo
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)

    # resample the signal if the sampling rate is not 44100
    if sampling_rate != sr:
        signal = resample_sig(signal, original_sr=sr, target_sr=sampling_rate)

    # normalize the signal to the custom range
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    signal = signal * (n_range[1] - n_range[0]) + n_range[0]

    return signal


def audio_loader(path, sr=22500, n_workers=min(mp.cpu_count(), 4), norm_range = (-1, 1)):
    """Loads the audio data from a directory and returns the data
    as a pandas Dataframe.

    The directory should:
        - contain subdirectories containing the wav files
            - the name of the subdirectories should be the label
            of the wav files.
        or
        - contain wav files itself
            - the name of the directory will be the label of the wav files.

    Args:
        path (str): path to the directory
        sr (int, optional): sampling rate of the audio data. Defaults to 44100.
        n_workers (int, optional): number of workers for multiprocessing.
                                    Defaults to mp.cpu_count().
        norm_range (tuple, optional): range for normalization. Defaults to (-1, 1).

    Returns:
        pandas Dataframe: data from the wav files in a pandas Dataframe.
     """

    data = []  # sound data
    df = []  # dataframes
    subdirs, subdirnames = get_sub_dirs(path)

    global sampling_rate
    global n_range
    
    n_range = copy.deepcopy(norm_range)
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
