from pprint import pprint
import os
import sys
import time

import librosa

import numpy as np
import toml

from sklearn import pipeline

sys.path.append(f"{os.getcwd()}")

try:
    from crossai.processing import Audio
    from crossai.processing import Motion
    from crossai.processing import TimeSeries
    from crossai.preparation import Transformations
    from crossai.preparation import Scaler
    from crossai.preparation import Encoder
except ModuleNotFoundError:
    raise "Cannot find audio module"

CONFIG = toml.load(os.getcwd() + '/examples/config_default.toml')
if __name__ == "__main__":

    audio_instance = librosa.load(os.getcwd() + '/examples/test_audio.wav', sr=44100)[0]


    # For testing
    audio_data = [
        {
            'X': [audio_instance, audio_instance],
            'Y': "Crackle"

        },
        {
            'X': [audio_instance, audio_instance],
            'Y': "Crackle"

        },
        # {
        #     'X': [np.arange(len(audio_instance)-20), np.arange(len(audio_instance)-20)],
        #     'Y': "Wheeze"

        # },
        # {
        #     'X': np.arange(len(audio_instance)-10),
        #     'Y': "Crackle"

        # },
        # {
        #     'X': audio_instance,
        #     'Y': "Crackle"

        # },
        # {
        #     'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
        # {
        #     'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
        # {
        #     'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
        # {
        #     'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
        # {
        #     'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
    ]

    motion_data = [
        {
            "X": {

                "acc_x":np.arange(2000)/20,
                "acc_y":np.arange(2000),
                "acc_z":np.arange(2000),
                "gyr_x":np.arange(2000),
                "gyr_y":np.arange(2000),
                "gyr_z":np.arange(2000)

            },
            "Y": 3
        },
        {
            "X": {

                "acc_x":np.arange(200),
                "acc_y":np.arange(2000),
                "acc_z":np.arange(2000),
                "gyr_x":np.arange(2000),
                "gyr_y":np.arange(2000),
                "gyr_z":np.arange(2000)

            },
            "Y": 3
        },
        # {
        #     "X": {

        #         "acc_x":[np.arange(2000)],
        #         "acc_y":[np.arange(2000)],
        #         "acc_z":[np.arange(2000)],
        #         "gyr_x":[np.arange(2000)],
        #         "gyr_y":[np.arange(2000)],
        #         "gyr_z":[np.arange(2000)]

        #     },
        #     "Y": 3
        # },
        # {
        #     "X": {

        #         "acc_x":np.arange(2000)/11,
        #         "acc_y":np.arange(2000)/13,
        #         "acc_z":np.arange(2000)/10,
        #         "gyr_x":np.arange(2000)/10,
        #         "gyr_y":np.arange(2000)/10,
        #         "gyr_z":np.arange(2000)

        #     },
        #     "Y": 1
        # },
        # {
        #     "X": {

        #         "acc_x":np.arange(2000)/11,
        #         "acc_y":np.arange(2000)/13,
        #         "acc_z":np.arange(2000)/10,
        #         "gyr_x":np.arange(2000)/10,
        #         "gyr_y":np.arange(2000)/10,
        #         "gyr_z":np.arange(2000)

        #     },
        #     "Y": 1
        # },
        # {
        #     "X": {

        #         "acc_x":np.arange(999)/11,
        #         "acc_y":np.arange(1000)/13,
        #         "acc_z":np.arange(1000)/10,
        #         "gyr_x":np.arange(1000)/10,
        #         "gyr_y":np.arange(1000)/10,
        #         "gyr_z":np.arange(1000)

        #     },
        #     "Y": 1
        # },
    ]

    motion = Motion(motion_data)
    audio = Audio(audio_data)
    ts = TimeSeries(audio_data)

    # pprint.pprint(CONFIG)
    # print('\n\n')
    transformers = motion.get_transformers(CONFIG['processing'])
    t = Transformations(1).transformers(CONFIG['transformation'])
    scalers = Scaler(scale_axis=2).transformers(CONFIG['scaler'])
    encoders = Encoder().transformers(CONFIG['encoder'])

    start = time.monotonic()
    pipe = pipeline.Pipeline([
        ('passthrough', None),
        # ('mono', transformers['convert_to_mono']),
        # ('filter', transformers['butterworth_filter']),
        # ('savgol', transformers['savgol_filter']),
        # ('sliding_window', transformers['sliding_window']),
        # ('pad', transformers['pad']),
        # ('abs', transformers['abs']),
        # ('accel', transformers['pure_acceleration']),
        # ('mean', transformers['mean']),
        # ('cross', transformers['cross_correlation']),
        # ('stft', transformers['stft']),
        # ('energy', transformers['energy']),
        ('loud', transformers['spectral_entropy']),
        # ('istft', transformers['istft']),
        # ('mfcc', transformers['mfccs']),
        # ('mel', transformers['melspectrogram']),
        # ('real', transformers['complex_to_real']),
        # ("xy", transformers['get_features']),
        ("xy", transformers['get_data']),
    ])
    end = time.monotonic()
    pipe2 = pipeline.Pipeline([
        ("pipe1", pipe),
        ("split", t['split']),
        ("pca", t['lda']),
        # ("scale", scalers['standard_scaler']),
        # ("onehot", encoders['one_hot_encoder']),
    ])

    
    f = pipe2.fit_transform(None)
    print(f)
    # audio._pretty_print_data
    # audio._pretty_print_features
    # print(motion.data)
    # print(f.shape)