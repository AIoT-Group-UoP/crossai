import numpy as np
import librosa
import cv2
from matchering import Config
from matchering.stages import main
from matchering.saver import save
from matchering.utils import get_temp_folder
from matchering import pcm24


def spectrum_calibration(
    target: str,
    reference: str,
    config: Config = Config(),
    save_to_wav: bool = False,
    to_mono: bool = True
) -> np.ndarray:
    """Processes the target audio to match the reference audio.

    Args:
        target: Target audio.
        reference: Reference audio.
        config: Configuration for the processing.
        save_to_wav: Whether to save the processed audio to a wav file.
        to_mono: Whether to convert the processed audio to mono.

    Returns:
        correct_result: Processed audio.
    """
    results = [pcm24("result.wav")]
    # Get a temporary folder for converting mp3's
    temp_folder = config.temp_folder if config.temp_folder \
        else get_temp_folder(results)

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


def q_transform(
    sig: np.ndarray,
    sr: int = 44100,
    n_bins: int = 84,
    hop_length: int = 512,
    fmin: int = 55,
    norm: int = 1,
    bins_per_octave: int = 12,
    tuning: int = None,
    filter_scale: int = 1,
    sparsity: int = 0.01,
    window: str = 'hann',
    scale: bool = True,
    pad_mode: str = 'reflect',
    to_db: bool = True,
    dsize: tuple = None
) -> np.ndarray:
    """Computes the constant-Q transform of an audio signal.

    Args:
        sig: Input signal.
        sr: Sampling rate of the input signal.
        n_bins: Number of frequency bins.
        hop_length: Number of samples between successive frames.
        fmin: Minimum frequency.
        norm: Normalization factor.
        bins_per_octave: Number of bins per octave.
        tuning: Deviation from A440 tuning in fractional bins.
        filter_scale: Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
        sparsity: Sparsity of the CQT basis.
        window: Type of window to use.
        scale: If True, scale the magnitude of the CQT by n_bins
        pad_mode: If center=True, the padding mode to use at the edges of the signal. By default, STFT uses reflection padding.
        to_db: Convert the spectrogram to dB scale.
        dsize: Size of the output spectrogram : if None, the output is the raw spectrogram.

    Returns:
        q_transform: Returns the constant-Q transform of an audio signal.
    """

    q_transform = librosa.core.cqt(sig, sr=sr, n_bins=n_bins,
                                   hop_length=hop_length, fmin=fmin,
                                   norm=norm, bins_per_octave=bins_per_octave,
                                   tuning=tuning, filter_scale=filter_scale,
                                   sparsity=sparsity, window=window,
                                   scale=scale, pad_mode=pad_mode)

    if to_db is True:
        q_transform = librosa.amplitude_to_db(q_transform, ref=np.max)

    if dsize is not None:
        q_transform = cv2.resize(
            q_transform, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return q_transform


def melspectrogram(
    sig: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: int = 0.0,
    fmax: int = 8000,
    power: int = 2,
    to_db: bool = True,
    dsize: tuple = None
) -> np.ndarray:
    """Computes a mel-scaled spectrogram.

    Args:
        sig: Input signal.
        sr: Sampling rate of the input signal.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames.
        n_mels: Number of Mel bands to generate.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        power: Power of the spectrogram.
        to_db: Convert the spectrogram to dB scale.
        dsize: Size of the output spectrogram : if None, the output is the raw spectrogram.

    Returns:
        mel_spectrogram: Returns a mel-scaled spectrogram.
    """

    mel_spectrogram = librosa.feature.melspectrogram(
        y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax, power=power)

    if to_db is True:
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if dsize is not None:
        mel_spectrogram = cv2.resize(
            mel_spectrogram, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return mel_spectrogram


def inverse_melspectrogram(
    sig: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'reflect',
    power: float = 2.0,
    n_iter: int = 32,
    length: int = None
) -> np.ndarray:
    """Computes the inverse of a mel-scaled spectrogram.

    Args:
        sig: Input signal.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames.
        win_length: Each frame of audio is windowed by window of length win_length and then padded with zeroes to match n_fft.
        window: Type of window to use.
        center: If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].
        pad_mode: If center=True, the padding mode to use at the edges of the signal. By default, STFT uses reflection padding.
        power: Power of the spectrogram.
        n_iter:  Number of inversion iterations.
        length: If provided, the output y is zero-padded or clipped to exactly length samples.

    Returns:
        inverse_mel_spectrogram : Returns the inverse of a mel-scaled spectrogram.
    """

    inverse_mel_spectrogram = librosa.feature.inverse.mel_to_audio(
        M=sig, sr=44100, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, center=center,
        pad_mode=pad_mode, power=power, n_iter=n_iter, length=length)

    return inverse_mel_spectrogram


def chroma(
    sig: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    fmin: float = None,
    norm: int = 1,
    threshold: float = 0.0,
    tuning: float = None,
    n_chroma: int = 12,
    n_octaves: int = 7,
    window: str = None,
    bins_per_octave: int = 36,
    cqt_mode: str = 'full',
    dsize: tuple = None
) -> np.ndarray:
    """Computes a chromagram from a waveform or power spectrogram.

    Args:
        sig: Input signal.
        sr: Sampling rate of the input signal.
        hop_length: Number of samples between successive frames.
        fmin: Minimum frequency.
        norm: Normalization factor.
        threshold: Pre-normalization energy threshold. Values below the threshold are discarded, resulting in a sparse chromagram.
        tuning: Deviation from A440 tuning in fractional bins.
        n_chroma: Number of chroma bins to produce.
        n_octaves: Number of octaves to analyze above fmin.
        window: Type of window to use.
        bins_per_octave: Number of bins per octave.
        cqt_mode: Constant-Q transform mode.
        dsize: Size of the output spectrogram : if None, the output is the raw spectrogram.

    Returns:
        chroma: Returns the chromagram for an audio signal
    """

    chroma = librosa.feature.chroma_cqt(y=sig, sr=sr, hop_length=hop_length,
                                        fmin=fmin, norm=norm,
                                        threshold=threshold, tuning=tuning,
                                        n_chroma=n_chroma, n_octaves=n_octaves,
                                        window=window,
                                        bins_per_octave=bins_per_octave,
                                        cqt_mode=cqt_mode)

    if dsize is not None:
        chroma = cv2.resize(
            chroma, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return chroma


def chroma_cens(
    sig: np.ndarray,
    sr: int,
    n_chroma: int = 12,
    hop_length: int = 512,
    fmin: float = None,
    norm: int = 1,
    tuning: float = None,
    n_octaves: int = 7,
    bins_per_octave: int = 36,
    cqt_mode: str = 'full',
    dsize: tuple = None
) -> np.ndarray:
    """Computes the chroma variant “Chroma Energy Normalized” (CENS).

    Args:
        sig: Input signal.
        sr: Sampling rate of the input signal.
        n_chroma: Number of chroma bins to produce.
        hop_length: Number of samples between successive frames.
        fmin: Minimum frequency.
        norm: Normalization factor.
        tuning: Deviation from A440 tuning in fractional bins.
        n_octaves: Number of octaves to analyze above fmin.
        bins_per_octave: Number of bins per octave.
        cqt_mode: Constant-Q transform mode.
        dsize: Size of the output spectrogram : if None, the output is the raw spectrogram.

    Returns:
        chroma_cens: Returns the chroma variant (CENS).
    """

    chroma_cens = librosa.feature.chroma_cens(y=sig, sr=sr, n_chroma=n_chroma,
                                              hop_length=hop_length, fmin=fmin,
                                              norm=norm, tuning=tuning,
                                              n_octaves=n_octaves,
                                              bins_per_octave=bins_per_octave,
                                              cqt_mode=cqt_mode)

    if dsize is not None:
        chroma_cens = cv2.resize(
            chroma_cens, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return chroma_cens


def chroma_stft(
    sig: np.ndaarray,
    sr: int,
    n_chroma: int = 12,
    hop_length: int = 512,
    win_length: int = None,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'reflect',
    tuning: float = None,
    dsize: tuple = None
) -> np.ndarray:
    """Computes a  stft chromagram from a waveform or power spectrogram.

    Args:
        sig: Input
        sr: Sampling rate of the input signal.
        n_chroma: Number of chroma bins to produce.
        hop_length: Number of samples between successive frames.
        win_length: Each frame of audio is windowed by window().The window will be of length win_length and then padded with zeros to match n_fft.
        window: Type of window to use.
        center: If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].
        pad_mode: If center=True, the padding mode to use at the edges of the signal. By default, STFT uses reflection padding.
        tuning: Deviation from A440 tuning in fractional bins.
        dsize: Size of the output spectrogram : if None, the output is the raw spectrogram.

    Returns:
        chroma_stft: Returns the chromagram for an audio signal.
    """

    chroma_stft = librosa.feature.chroma_stft(y=sig, sr=sr, n_chroma=n_chroma,
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              window=window, center=center,
                                              pad_mode=pad_mode, tuning=tuning)

    if dsize is not None:
        chroma_stft = cv2.resize(
            chroma_stft, dsize=dsize, interpolation=cv2.INTER_CUBIC)

    return chroma_stft


def mfcc(
    sig: np.ndarray,
    sr: int = 22050,
    spec: np.ndarray = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: str = 'ortho',
    lifter: int = 0
) -> np.ndarray:
    """
    Compute the MFCCs (Mel-frequency cepstral coefficients) from an audio signal.

    Args:
        sig: Input signal.
        sr: Sampling rate of the input signal.
        spec: Pre-computed spectrogram.
        n_mfcc: Number of MFCCs to return.
        dct_type: Type of DCT (discrete cosine transform) to use.
        norm: Type of norm to use.
        lifter: Parameter for inversion of MFCCs.

    Returns:
        mfcc: Mel-frequency cepstral coefficients.
    """

    mfcc = librosa.feature.mfcc(
        y=sig, sr=sr, S=spec, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm,
        lifter=lifter)

    return mfcc
