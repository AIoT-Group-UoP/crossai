import numpy as np
import librosa


def spectral_skewness(x: np.ndarray):
    """
    Calculate the spectral skewness of a audio signal.
    """
    # Calculate the power spectrum
    x = x.astype(float)
    spectrum = np.abs(np.fft.fft(x))**2
    # Sum across all frequency bins
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    skew = np.mean(np.power(spectrum - mean, 3)) / (std ** 3)
    return skew


def spectral_kurtosis(x: np.ndarray):
    """
    Calculate the spectral skewness of a audio signal.
    """
    x = x.astype(float)
    # Calculate the power spectrum
    spectrum = np.abs(np.fft.fft(x))**2
    # Sum across all frequency bins
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    kurtosis = np.mean(np.power(spectrum - mean, 4)) / (std ** 4)
    return kurtosis


def loudness(x: np.ndarray) -> float:
    """Caclulate the loudness of a audio signal.

    Args:
        x (np.ndarray): Input signal.

    Returns:
        float: Loudness
    """
    rms = np.sqrt(np.mean(x**2))
    loudness = 20 * np.log10(rms) # to db
    return  loudness


def melspectogram_power_filtering(x : np.ndarray, Fs : int, n_fft : int, hop_length : int, win_length : int, window : str, center : bool , pad_mode : str , fmax : int, power_signal : float, power_noise : float):
    """Remove noise from a melspectogram.
    
    Args:
        x (np.ndarray): Input signal.
        sr (int): Sampling rate.
        fmin (int): Minimum frequency.
        fmax (int): Maximum frequency.
        n_mels (int): Number of mel bands.
        power_signal (float): Power of the signal.
        power_noise (float): Power of the noise.

    Returns:
        np.ndarray: Melspectogram with noise removed.    
    """
    
    noise = librosa.feature.melspectrogram(x, sr=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, fmax=fmax, power = power_noise)
    spectogram = librosa.feature.melspectrogram(x, sr=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, fmax=fmax, power = power_signal)
    
    return librosa.power_to_db(spectogram - noise)


def logmelspectogram(x : np.ndarray, Fs : int, n_fft : int, hop_length : int, win_length : int, window : str, center : bool , pad_mode : str , power : float, fmax : int):
    """ Calculate the logmelspectogram of a audio signal.
    
    Args:
            Fs (float): Sampling rate.
            n_fft (int): Number of FFTs.
            hop_length (int): Window hop size.
            win_length (int): Window's size.
            window (str): Window type.
            center (bool): Centering the frame.
            pad_mode (str): Pad at the end.
            power (float): Exponent for the magnitude melspectrogram.
            fmax (float): Maximum frequency.
        
    """
    
    melspectrogram = librosa.feature.melspectrogram(x, sr=Fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, power=power, fmax=fmax)
    
    return librosa.power_to_db(melspectrogram)