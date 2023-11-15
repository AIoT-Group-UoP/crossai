import numpy as np
import librosa
import cv2
from matchering.log import Code, info, debug, debug_line, ModuleError
from matchering import Config, Result
from matchering.loader import load
from matchering.stages import main
from matchering.saver import save
from matchering.preview_creator import create_preview
from matchering.utils import get_temp_folder
from matchering.checker import check, check_equality
from matchering.dsp import channel_count, size
from matchering import pcm24


def spectrum_calibration(target: str, reference: str, sr: float, config: Config = Config(), save_to_wav=False, to_mono=True):
    """
    Processes the target audio to match the reference audio.

    Parameters
        target: target audio
        reference: reference audio
        sr: sample rate of the audio
        config: configuration for the processing
        save_to_wav: whether to save the processed audio to a wav file
        to_mono: whether to convert the processed audio to mono

    Returns
        correct_result: processed audio
    """
    results = [pcm24("result.wav")]
    # Get a temporary folder for converting mp3's
    temp_folder = config.temp_folder if config.temp_folder else get_temp_folder(
        results)

    target = np.vstack((target, target)).T
    reference = np.vstack((reference, reference)).T

    # Process
    result, result_no_limiter, result_no_limiter_normalized = main(
        target,
        reference,
        config,
        need_default=any(rr.use_limiter for rr in results),
        need_no_limiter=any(
            not rr.use_limiter and not rr.normalize for rr in results),
        need_no_limiter_normalized=any(
            not rr.use_limiter and rr.normalize for rr in results
        ),
    )
    del reference
    del target

    # Output
    for required_result in results:
        if required_result.use_limiter:
            correct_result = result

        else:
            if required_result.normalize:
                correct_result = result_no_limiter_normalized
            else:
                correct_result = result_no_limiter

        if save_to_wav:
            save(
                required_result.file,
                correct_result,
                config.internal_sample_rate,
                required_result.subtype,
            )
        # convert to mono if needed
        if to_mono:
            correct_result = correct_result.mean(axis=1)

        return correct_result


def q_transform(sig, sr=44100, n_bins=84, hop_length=512, fmin=55, norm=1,
                bins_per_octave=12, tuning=None, filter_scale=1, sparsity=0.01,
                window='hann', scale=True, pad_mode='reflect', to_dB=True,
                dsize=None):
    """
    Compute the constant-Q transform of an audio signal.

    Args:
        sig (numpy array): Input signal
        sr (int): Sampling rate of the input signal
        n_bins (int): Number of frequency bins
        hop_length (int): Number of samples between successive frames
        fmin (float): Minimum frequency
        norm (int): Normalization factor
        bins_per_octave (int): Number of bins per octave
        tuning (float): Deviation from A440 tuning in fractional bins
        filter_scale (float): Filter scale factor. Small values (<1) use
                                shorter windows for improved time resolution.
        sparsity (float): Sparsity of the CQT basis
        window (string): Type of window to use
        scale (bool): If True, scale the magnitude of the CQT by n_bins
        pad_mode (string): If center=True, the padding mode to use at the
                            edges of the signal. 
                                By default, STFT uses reflection padding.
        to_dB (bool): Convert the spectrogram to dB scale
        dsize (tuple): Size of the output spectrogram : if None, the output is the raw spectrogram 

    Returns:
        q_transform (numpy array): Returns the constant-Q transform of an
                                    audio signal
    """

    q_transform = librosa.core.cqt(sig, sr=sr, n_bins=n_bins,
                                   hop_length=hop_length, fmin=fmin,
                                   norm=norm, bins_per_octave=bins_per_octave,
                                   tuning=tuning, filter_scale=filter_scale,
                                   sparsity=sparsity, window=window,
                                   scale=scale, pad_mode=pad_mode)

    if to_dB is True:
        q_transform = librosa.amplitude_to_db(q_transform, ref=np.max)

    if dsize is not None:
        q_transform = cv2.resize(
            q_transform, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return q_transform
