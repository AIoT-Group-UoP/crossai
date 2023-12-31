import boto3
import copy
import wave
import io
import multiprocessing as mp
import pandas as pd
import numpy as np
from collections import defaultdict
from crossai.processing import resample_sig


def s3_wavfile_reader(file_content):
    """Reads a wav file from a byte stream and returns data as numpy array.

    Args:
        file_content (bytes): Byte content of the wav file.

    Returns:
        numpy array: Data from the wav file.
    """

    # Create a file object from bytes
    file_obj = io.BytesIO(file_content)

    # Open the file object in wave module
    wav_file = wave.open(file_obj, "rb")

    # Read frames and convert to byte array
    signal = np.frombuffer(
        wav_file.readframes(wav_file.getnframes()), dtype=np.int16
    )

    # Convert to float32
    signal = signal.astype(np.float32)

    # Get the sample rate
    sr = wav_file.getframerate()

    # resample the signal if the sampling rate is not 44100
    if sampling_rate != sr:
        signal = resample_sig(signal, original_sr=sr, target_sr=sampling_rate)

    # normalize the signal
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    return signal


def s3_audio_loader(
    bucket,
    prefix="",
    endpoint_url="https://s3.amazonaws.com",
    sr=22500,
    n_workers=min(mp.cpu_count(), 4),
):
    """Loads audio files from s3 bucket and returns a pandas dataframe.

    The directory should:
        - contain subdirectories containing the wav files
        - the name of the subdirectories should be the label
            of the wav files.
        or
        - contain wav files itself
        - the name of the directory will be the label of the wav files.

    Args:
        bucket (str): name of the s3 bucket
        prefix (str, optional): prefix of the directory. Defaults to "".
        endpoint_url (str, optional): endpoint url of the s3 bucket.
                                    Defaults to "https://s3.amazonaws.com".
        sr (int, optional): sampling rate of the audio data.
        n_workers (int, optional): number of workers for multiprocessing.
                                    Defaults to mp.cpu_count().

    Returns:
        pandas Dataframe: data from the wav files in a pandas Dataframe.
    """

    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket, Prefix=prefix, Delimiter="/"
    )

    data = []
    df = []
    subdirnames = []

    global sampling_rate
    sampling_rate = copy.deepcopy(sr)

    label_counter = defaultdict(int)  # counter for each unique label

    # load the sound data using multiprocessing
    for page in page_iterator:
        for prefix in page.get("CommonPrefixes", []):
            subdir_prefix = prefix["Prefix"]
            subdirnames.append(subdir_prefix)
            subdir_page_iterator = paginator.paginate(
                Bucket=bucket, Prefix=subdir_prefix
            )
            for subdir_page in subdir_page_iterator:
                keys = [obj["Key"] for obj in subdir_page["Contents"]]
                pool = mp.get_context("fork").Pool(n_workers)
                data.append(
                    pool.map(
                        s3_wavfile_reader,
                        [
                            s3.get_object(Bucket=bucket, Key=key)[
                                "Body"
                            ].read()
                            for key in keys
                        ],
                    )
                )
                pool.close()
                pool.join()

    for i in range(len(data)):
        for j in range(len(data[i])):
            # Extract the parent folder name from the key
            # and use it as the label
            label = keys[j].split("/")[-2]
            data[i][j] = (data[i][j].astype(np.float32), label)

    df = pd.DataFrame(columns=["data", "label", "indice"])

    for i in range(len(data)):
        for j in range(len(data[i])):
            label = data[i][j][1]
            df.loc[len(df)] = [data[i][j][0], label, label_counter[label]]
            label_counter[label] += 1  # increment the counter for the label

    return df
