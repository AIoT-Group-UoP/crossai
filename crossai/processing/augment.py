import random
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


def spec_augment(spectrogram, augment_times=1, masks=2, freq_masking=0.15, time_masking=0.15):
    """
    Implementation of SpecAugment using numpy.

    Args:
        spectrogram (numpy.ndarray): Input 2D spectrogram with shape (freq, time).
        augment_times (int): Number of times to augment original spectrogram.
        masks (int): Number of masks for frequency and time masking.
        freq_masking (float, optional): Maximum frequency masking length. Defaults to 0.15.
        time_masking (float, optional): Maximum time masking length. Defaults to 0.15.

    Returns:
        augmented_spectrograms (numpy.ndarray): List of augmented spectrograms.
    """

    augmented_spectrograms = []
    original_spectrogram = spectrogram.copy()
    masks = max(1, masks)  # mask cannot be 0 or negative

    for i in range(augment_times):
        augmented = original_spectrogram.copy()

        for i in range(masks):

            # frequency masking
            freqs, time_frames = augmented.shape
            freq_mask_percentage = random.uniform(0.0, freq_masking)
            masked_freqs = int(freq_mask_percentage * freqs)
            f0 = int(np.random.uniform(low=0.0, high=freqs - masked_freqs))
            augmented[f0:f0 + masked_freqs, :] = spectrogram.min()

            # time masking
            time_frames_mask_percentage = random.uniform(0.0, time_masking)
            masked_time_frames = int(time_frames_mask_percentage * time_frames)
            t0 = int(np.random.uniform(
                low=0.0, high=time_frames - masked_time_frames))
            augmented[:, t0:t0 + masked_time_frames] = spectrogram.min()

        augmented_spectrograms.append(augmented)

    return augmented_spectrograms


def add_noise(signal, augment_times, noise_factor=0.005):
    """
    Add random noise to the signal

    Args:
        signal (numpy array): Input signal
        sr (int): Sampling rate of the input signal
        noise_factor (float): Noise factor

    Returns:
        noisy_signal (numpy array): Returns the noisy signal
    """

    noisy_signal = []
    for i in range(augment_times):
        noise = np.random.randn(len(signal))
        noisy_signal.append(signal + noise_factor*noise)
    return noisy_signal
