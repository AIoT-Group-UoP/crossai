from typing import Union
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import albumentations as S
import audiomentations as A


class SpecAugment:

    def __init__(self, num_augment) -> None:

        self.__augment_ops = [] 
        self.__augmenter = None
        self.num_augment = num_augment


    def __call__(self, config):
        for k, v in config.items():
            if k in dir(self):
                self.__getattribute__(k)(**v)
                self.__augmenter = S.Compose(tuple(self.__augment_ops))
        return FunctionTransformer(self.augment_data)


    def brightness(
            self, 
            *, 
            limit: Union[float, tuple],
            always_apply: bool,
            p: float
    ):
        __op = S.RandomBrightness(limit=limit, always_apply=always_apply, p=p)
        self.__augment_ops.append(__op)
        
    def median_blur(
            self,
            *,
            blur_limit: float,
            always_apply: bool,
            p: float
    ):
        __op = S.MedianBlur(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.__augment_ops.append(__op)

    def blur(
            self,
            *,
            blur_limit: float,
            always_apply: bool,
            p: float
    ):
        __op = S.Blur(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.__augment_ops.append(__op)

    def pixel_dropout(
            self,
            *,
            dropout_prob: float,
            always_apply: bool,
            p: float,
            mask_drop_value: Union[list, float] = None,
            drop_value: Union[list, float] = 0.0,
            per_channel: bool = False
    ):
        __op = S.PixelDropout(dropout_prob=dropout_prob, always_apply=always_apply, p=p,
                              mask_drop_value=mask_drop_value, drop_value=drop_value, per_channel=per_channel)
        self.__augment_ops.append(__op)

    def augment_data(self,data):
        data = list(data)
        x = data[0]
        if len(data) == 6:
            y = data[3]
        elif len(data) == 4:
            y = data[2]

        _new_data = []
        _y = []
        for instance in range(len(x)):
            _current = np.asarray(x[instance], dtype=np.float32)
            _current_shape = _current.shape
            _current_max = np.max(np.abs(_current))
            _new_data.append(_current)
            _y.append(y[instance])
            _current /= _current_max
            if len(_current_shape) == 2:
                _current = np.expand_dims(_current, 2)
            elif len(_current_shape) == 3:
                _current = np.swapaxes(_current, 0, -1)
            for _ in range(self.num_augment):
                _aug = self.__augmenter(image=_current)['image'] * _current_max
                if len(_current_shape) == 2:
                      _aug = _aug.reshape(_aug.shape[:-1])
                elif len(_current_shape) == 3:
                      _aug = np.swapaxes(_aug, 0, -1)
                _new_data.append(_aug)
                _y.append(y[instance])

        data[0] = np.array(_new_data, dtype=np.float32)
        _y = np.array(_y)
        if len(data) == 6:
            data[3] = _y
        elif len(data) == 4:
            data[2] = _y
        return tuple(data)
