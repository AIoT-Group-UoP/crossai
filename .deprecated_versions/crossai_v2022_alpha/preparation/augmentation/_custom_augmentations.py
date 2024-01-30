from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from tsaug._augmenter.base import _Augmenter, _default_seed
import nlpaug.augmenter.audio as naa

"""
Costum Augmentation module for audio data.

Each Augmentation Class represents a single augmentation operation on audio data and is a subclass of _Augmenter class.

Every Augmentation Class must contain the following methods:
    __init__ : initializes the class
    _get_param_name : returns the name of the class
    _augment_core : performs the augmentation operation on the input data

To add a new augmentation operation, create a new class that inherits from _Augmenter 
and implements the above methods. 
"""


class Vtlp_Aug(_Augmenter):
    """
    Augmenter that applies vocal tract length perturbation (VTLP) operation to audio.
    """

    def __init__(
            self,
            sr: int,
            repeats: int = 1,
            prob: float = 1.0,
            seed: Optional[int] = _default_seed,
    ):
        self.sr = sr
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("Vtlp_Aug",)

    def _augment_core(
            self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Y_aug = Y
        X_aug = []
        shape0 = X.shape[0]
        shape2 = X.shape[2]

        # reshape X to 2D array
        X = X.reshape(shape0, -1).astype(float)

        # apply vtlp augmentation to each row of X
        for i in range(shape0):
            X_aug.append(naa.VtlpAug(sampling_rate=self.sr).augment(X[i]))
        X_aug = np.asarray(X_aug)

        # pad zeros to the third axis of X_aug to match X shape
        X_aug = np.append(X_aug, np.zeros((X.shape[0], X_aug.shape[1], X.shape[1] - X_aug.shape[2])), axis=2)

        X_aug = X_aug.reshape(shape0, X_aug.shape[2], shape2)

        return X_aug, Y_aug


class Pitch_Aug(_Augmenter):
    """
    Augmenter that applies pitch operation to audio.
    """

    def __init__(
            self,
            sr: int,
            factor: tuple,
            repeats: int = 1,
            prob: float = 1.0,
            n_times: int = 1,
            seed: Optional[int] = _default_seed,
    ):
        self.sr = sr
        self.factor = factor
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("Pitch_Aug",)

    def _augment_core(
            self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Y_aug = Y
        X_aug = []
        shape0 = X.shape[0]
        shape2 = X.shape[2]

        # reshape X to 2D array
        X = X.reshape(shape0, -1).astype(float)

        factor = tuple(self.factor)

        # apply pitch augmentation to each row of X
        for i in range(shape0):
            X_aug.append(naa.PitchAug(sampling_rate=self.sr, factor=factor).augment(X[i]))
        X_aug = np.asarray(X_aug)

        X_aug = X_aug.reshape(shape0, X_aug.shape[2], shape2)

        return X_aug, Y_aug


class Roll_Aug(_Augmenter):
    """
    Augmenter that applies roll operation to audio.
    """

    def __init__(
            self,
            repeats: int = 1,
            prob: float = 1.0,
            seed: Optional[int] = _default_seed,
    ):
        super().__init__(repeats=repeats, prob=prob, seed=seed)

    @classmethod
    def _get_param_name(cls) -> Tuple[str, ...]:
        return ("Roll_Aug",)

    def _augment_core(
            self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Y_aug = Y
        X_aug = []
        shape0 = X.shape[0]
        shape2 = X.shape[2]

        # reshape X to 2D array
        X = X.reshape(shape0, -1).astype(float)

        # apply roll augmentation to each row of X
        for i in range(shape0):
            pivot = np.random.randint(X[i].shape[0])
            X_aug.append(np.roll(X[i], pivot, axis=0))
        X_aug = np.asarray(X_aug)

        X_aug = X_aug.reshape(shape0, X_aug.shape[1], shape2)

        return X_aug, Y_aug
