import os
import sys
sys.path.append(f"{os.getcwd()}")
import tensorflow as tf

from crossai.models.nn2d import Xception, ResNET, Inception, VGG16


m = VGG16(
    input_shape=(75,75,3), 
    num_classes=3,
    pooling_type='max',
    dense_layers=2,
    dropout=True,
    dropout_first=False,
    dropout_rate=[0.8, 0.4],
    dense_units=[128, 64]
)
m.build((None, 75,75,3))
print(m.summary(expand_nested=True))