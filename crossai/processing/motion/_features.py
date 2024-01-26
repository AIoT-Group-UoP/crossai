import numpy as np
from scipy import signal


def pure_acceleration(Fs, acc_x, acc_y, acc_z, order=2, fc=1):
    """Extract the pure acceleration from the data using a
    high-pass filter (removing very low frequency drifts
    or motion effects).


    Args:
        Fs (int): Sampling frequency of the signal
        acc_x (numpy array): X dimension of input signal
        acc_y (numpy array): Y dimension of input signal
        acc_z (numpy array): Z dimension of input signal
        order (int): Order of the filter. Defaults to 2
        fc (int): Cut-off frequency of the filter. Defaults to 1

    Returns:
        pure_acc (list): List of pure acceleration
    """

    order = order  # Order of the filter
    fs = Fs
    fc = fc  # Cut-off frequency of the filter
    w = 2 * fc / fs

    sos = signal.butter(order, w, 'high', output='sos')

    pure_acc_x = signal.sosfilt(sos, acc_x)
    pure_acc_y = signal.sosfilt(sos, acc_y)
    pure_acc_z = signal.sosfilt(sos, acc_z)

    return pure_acc_x.astype(np.float32), \
        pure_acc_y.astype(np.float32), \
        pure_acc_z.astype(np.float32)
