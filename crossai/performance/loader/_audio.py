import numpy as np
import pandas as pd
from scipy.io import wavfile
from crossai.processing import resample_sig
from crossai.pipelines.audio import Audio
from crossai.performance import pilot_label_processing


def audio_loader(filename, classes, sampling_rate=22500):
    """ Loads an instance of audio (wav) data and returns the
    data in the equivalent crossai object for pilot evaluation

    Args:
        filename (str): path to the file
        classes (list): list of class names shaped [class1, class2, ...]
        sr (int, optional): sampling rate of the audio data. Defaults to 44100.

    Returns:
        crossai object: data in the equivalent crossai object
    """

    # get the sampling rate
    sr, signal = wavfile.read(filename)
    signal = signal.astype(np.float32)

    # resample the signal if the sampling rate is not the chosen one
    if sampling_rate != sr:
        signal = resample_sig(signal, original_sr=sr, target_sr=sampling_rate)

    # normalize the signal
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    labels = pilot_label_processing(filename.replace('.wav', '.json'),
                                    classes,
                                    len(signal),
                                    sampling_rate=sampling_rate)[0]
    # create the object
    df = pd.DataFrame(columns=['data', 'label', 'filename'])
    df['data'] = [signal]
    df['label'] = [labels]
    df['filename'] = [filename]

    crossai_object = Audio(df)

    return crossai_object
