from random import randint
import numpy as np


def roll_signal(sig, augment_times, n_roll=randint(10000, 50000)):
    """
    Roll the signal by n_roll samples

    Args:
        sig (numpy array): Input signal
        sr (int): Sampling rate of the input signal
        n_roll (int): Number of samples to roll

    Returns:
        rolled_sig (numpy array): Returns the rolled signal
    """
    rolled_sig = []
    for i in range(augment_times):
        rolled_sig.append(np.roll(sig, n_roll))
        n_roll = randint(20000, 50000)
    return rolled_sig
