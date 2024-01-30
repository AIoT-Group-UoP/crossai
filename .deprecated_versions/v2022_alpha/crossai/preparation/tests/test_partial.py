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

    audio_data = [

        { 
            "X" : [0, 0],
            "Y": 1
        },
        { 
            "X" : [0, 0],
            "Y": 1
        },

        { 
            "X" : [1, 1],
            "Y": 1
        },
        { 
            "X" : [1, 1],
            "Y": 1
        },


    ]


    audio = Audio(None)

    transformers = audio.get_transformers(CONFIG['processing'])
    t = Transformations(1).transformers(CONFIG['transformation'])
    scaler_obj = Scaler(scale_axis=0, partial=False)
    scalers = scaler_obj.transformers(CONFIG['scaler'])
    encoders = Encoder().transformers(CONFIG['encoder'])

    start = time.monotonic()
    pipe = pipeline.Pipeline([
        ('set_data', transformers['set_data']),
        ("xy", transformers['get_data']),
    ])
    end = time.monotonic()

    pipe2 = pipeline.Pipeline([
        ("pipe1", pipe),
        ("scale", scalers['standard_scaler']),
    ])

    pipe3 = pipeline.Pipeline([
        ("pipe1", pipe),
        ("trans", scalers['toggle_fit']),
        ("scale", scalers['standard_scaler']),
    ])
    
    for i in audio_data:
        f = pipe2.fit_transform([i])

    for i in audio_data:
        f = pipe3.fit_transform([i])
    print(scaler_obj.scaler.mean_)
    print(f)