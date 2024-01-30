from pprint import pprint
import os
import sys
import time
from scipy.io import wavfile

import librosa

import numpy as np
import toml

from sklearn import pipeline

sys.path.append(f"{os.getcwd()}")

try:
    from crossai.processing import Audio, Motion, TimeSeries
    from crossai.preparation import Transformations, Scaler, Encoder
except ModuleNotFoundError:
    raise "Cannot find audio module"

CONFIG = toml.load(os.getcwd() + '/examples/config_default.toml')

if __name__ == "__main__":

    audio_instance = wavfile.read(os.getcwd() + '/examples/test_audio.wav')[1]


    # For testing
    # eval_audio = [
    #     {
    #         'X': [audio_instance, audio_instance],
    #         'Y': "Crackle"

    #     },
    # ]
    audio_data = [
        # {
        #     'X': audio_instance,
        #     'Y': "Crackle"

        # },
        # {
        #     'X': audio_instance, 
        #     'Y': "Crackle"

        # },
        # {
        #     'X': np.arange(len(audio_instance)-20),
        #     'Y': "Wheeze"
        # },
        {
            'X': [np.arange(len(audio_instance)-1000), np.arange(len(audio_instance)-1000)],
            'Y': "Wheeze"

        },
        # {
        #     'X': audio_instance,
        #     'Y': "Crackle"

        # },
        {
            'X': [audio_instance, audio_instance],
            'Y': "Crackle"
        },
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

    eval_motion = [
        {
            "X": {

                "acc_x":np.arange(200)/20,
                "acc_y":np.arange(200),
                "acc_z":np.arange(200),
                "gyr_x":np.arange(200),
                "gyr_y":np.arange(200),
                "gyr_z":np.arange(200)

            },
            "Y": 3
        },
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

                "acc_x":np.arange(20),
                "acc_y":np.arange(20),
                "acc_z":np.arange(20),
                "gyr_x":np.arange(20),
                "gyr_y":np.arange(2000),
                "gyr_z":np.arange(2000)

            },
            "Y": 3
        },
        {
            "X": {

                "acc_x":[np.arange(2000)],
                "acc_y":[np.arange(2000)],
                "acc_z":[np.arange(2000)],
                "gyr_x":[np.arange(2000)],
                "gyr_y":[np.arange(2000)],
                "gyr_z":[np.arange(2000)]

            },
            "Y": 1
        },
        {
            "X": {

                "acc_x":np.arange(2000)/11,
                "acc_y":np.arange(2000)/13,
                "acc_z":np.arange(2000)/10,
                "gyr_x":np.arange(2000)/10,
                "gyr_y":np.arange(2000)/10,
                "gyr_z":np.arange(2000)

            },
            "Y": 1
        },
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

    motion = Motion(None)
    audio = Audio(audio_data)
    ts = TimeSeries(audio_data)

    # pprint.pprint(CONFIG)
    # print('\n\n')
    processing_t = audio.get_transformers(CONFIG['processing'])
    transformations_t = Transformations(1).transformers(CONFIG['transformation'])
    scaler_t = Scaler(scale_axis=1).transformers(CONFIG['scaler'])
    encoder_t = Encoder().transformers(CONFIG['encoder'])


    start = time.monotonic()
    pipe = pipeline.Pipeline([
        ('set_data',  processing_t['set_data']),
        ('crop', processing_t['crop']),
        # ('mono', processing_t['convert_to_mono']),
        # ('pad', processing_t['pad']),
        # ('sliding_window', processing_t['sliding_window']),
        # ('filter', processing_t['butterworth_filter']),
        # ('savgol', processing_t['savgol_filter']),
        # ('accel', processing_t['pure_acceleration']),
        # ('cross', processing_t['cross_correlation']),
        # ('stft', processing_t['stft']),
        # ('flux', processing_t['flux']),
        # ('real', processing_t['complex_to_real']),
        # ('energy', processing_t['energy']),
        # ('mean', processing_t['mean']),
        # ('istft', processing_t['istft']),
        # ('mfcc', processing_t['mfccs']),
        # ('mel', processing_t['melspectrogram']),
        ("xy", processing_t['get_data']),
        # ("split", transformations_t['split']),
        # ("pca", transformations_t['pca']),
        # ("scale", scaler_t['standard_scaler']),
        # ("split", transformations_t['split']),
        # ("onehot", encoder_t['one_hot_encoder']),
    ])

    end = time.monotonic()

    output = pipe.fit_transform(audio_data)
    # print(output[0])
    print(output[0].shape)

    # _idx = pipe.steps.index(('split', transformations_t['split']))

    # pipe.steps.remove(('split', transformations_t['split']))
    # pipe.steps.insert(_idx, ('extra', transformations_t['extra_data']))

    # print(pipe.fit_transform(eval_audio)[1])
    # if len(list(output)) == 6:
    #     x_train, x_val, x_test = output[:3]
    #     y_train, y_val, y_test = output[3:]
    # elif (len(list(output))) == 4:
    #     x_train, x_test = output[:2]
    #     y_train, y_test = output[2:]

    # print(output)
    # motion._pretty_print_data
    # audio._pretty_print_features