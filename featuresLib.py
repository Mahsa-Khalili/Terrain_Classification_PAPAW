import numpy as np

# For small float values
EPSILON = 0.00001


def l2norm(array):
    """L2 norm of an array"""
    return np.linalg.norm(array, ord=2)


def rms(array):
    """Root mean squared of an array"""
    return np.sqrt(np.mean(array ** 2))


def zcr(array):
    """Zero crossing rate of an array as a fraction of total size of array"""
    # divide by total datapoints in window
    return len(np.nonzero(np.diff(np.sign(array)))[0]) / len(array)


def msf(freqs, psd_amps):
    """Mean square frequency"""
    num = np.sum(np.multiply(np.resize(np.power(freqs, 2), len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)

    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON

    return np.divide(num, denom)


def rmsf(freqs, psd_amps):
    """Root mean square frequency"""
    return np.sqrt(msf(freqs, psd_amps))


def fc(freqs, psd_amps):
    """Frequency center"""
    num = np.sum(np.multiply(np.resize(freqs, len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)

    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON

    return np.divide(num, denom)


def vf(freqs, psd_amps):
    """Variance frequency"""
    return msf(freqs-fc(freqs, psd_amps), psd_amps)


def rvf(freqs, psd_amps):
    """Root variance frequency"""
    return np.sqrt(msf(freqs, psd_amps))
