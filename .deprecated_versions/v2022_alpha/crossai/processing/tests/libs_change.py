from pprint import pprint
import os
import sys
import time

import librosa
import librosa.display

import numpy as np
import toml

from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer

import matplotlib.pyplot as plt

sys.path.append(f"{os.getcwd()}")

from crossai.processing._utils import *

try:
    from crossai.processing import Audio, Motion, TimeSeries
    from crossai.preparation import Transformations, Scaler, Encoder
    from crossai.preparation.augmentation import Augmentation, SpecAugment
except ModuleNotFoundError:
    raise "Cannot find audio module"

CONFIG = toml.load(os.getcwd() + '/examples/config_default.toml')

if __name__ == "__main__":

    audio_instance = librosa.load(os.getcwd() + '/examples/test_audio.wav', sr=44100)[0]


    # For testing
    # eval_audio = [
    #     {
    #         'X': [audio_instance, audio_instance],
    #         'Y': "Crackle"

    #     },
    # ]
    audio_data = [
        # {
            # 'X': np.arange(100),
            # 'Y': "Crackle"
# 
        # },

        {   'X': np.arange(100),
            'Y': "Crackle"

        },
        {   'X': [audio_instance, audio_instance],
            'Y': "Crackle"

        },
        # {   'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
        # {   'X': [audio_instance, audio_instance],
        #     'Y': "Crackle"

        # },
        # {
        #     'X': [np.arange(len(audio_instance)-20), np.arange(len(audio_instance)-20)],
        #     'Y': "Crackle"
        # },
        # {
        #     'X': [np.arange(len(audio_instance)-10)],
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
    ]

    audio = Audio(None)

    # pprint.pprint(CONFIG)
    # print('\n\n')
    processing_t = audio.get_transformers(CONFIG['processing'])
    transformations_t = Transformations(1).transformers(CONFIG['transformation'])
    scaler_t = Scaler(scale_axis=1).transformers(CONFIG['scaler'])
    encoder_t = Encoder().transformers(CONFIG['encoder'])
    augment_t = SpecAugment(4)(CONFIG['augment'])


    sc = FunctionTransformer(concat_split)
    cs = FunctionTransformer(split_concated)

    start = time.monotonic()
    pipe = pipeline.Pipeline([
        ('set_data',  processing_t['set_data']),
        #('pad', processing_t['pad']),
        ('mono', processing_t['convert_to_mono']),
        ('mel', processing_t['melspectrogram']),
        # ('slide', processing_t['sliding_window']),
        # ('skewness', processing_t['spectral_skewness']),
        # ('kurtosis', processing_t['spectral_kurtosis']),
        # ('loudness', processing_t['loudness']),
        ("xy", processing_t['get_data']),
        ("split", transformations_t['split']),
        ("augme", augment_t),
        ("sc", sc),
        ('set_data1',  processing_t['set_data']),
        ('pad', processing_t['pad']),
        ("xy1", processing_t['get_data_dict']),
        ('cs', cs),
        ("scale", scaler_t['standard_scaler']),
        # ('mono', processing_t['convert_to_mono']),
        # ('pad', processing_t['pad']),
        # ('sliding_window', processing_t['sliding_window']),
        # ('savgol', processing_t['savgol_filter']),
        # ('accel', processing_t['pure_acceleration']),
        # ('mean', processing_t['mean']),
        # ('cross', processing_t['cross_correlation']),
        # ('stft', processing_t['stft']),
        # ('flux', processing_t['flux']),
        # ('real', processing_t['complex_to_real']),
        # ('energy', processing_t['energy']),
        # ('istft', processing_t['istft']),
        # ('mfcc', processing_t['mfccs']),
        # ("pca", transformations_t['pca']),
        # ("onehot", encoder_t['one_hot_encoder']),
    ])

    end = time.monotonic()

    d = pipe.fit_transform(audio_data)
    print(d[0].shape, d[1].shape)
    # fig, ax = plt.subplots()
    # librosa.display.specshow(d[7], sr=44100, ax=ax)
    # plt.savefig("spec.png")


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
    # audio._pretty_print_data
    # audio._pretty_print_features
