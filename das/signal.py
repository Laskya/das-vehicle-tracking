"""Signal preprocessing - denoising, filtering, normalisation, pooling."""

import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import maximum_filter

from .config import DT

POOLING_METHODS = ('max_pool', 'max_mean', 'median', 'median_max', 'local_max')


def denoise_das(df, wavelet='sym8', level=3, method='garrote'):
    """Wavelet shrinkage per channel, universal threshold with sigma from the MAD."""
    df_denoised = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        # copy=True matters - the column can be a read-only view into a
        # consolidated block and pywt needs something it can write to
        y = df[col].to_numpy(dtype=float, copy=True)
        coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
        approximation, details = coeffs[0], coeffs[1:]

        thresholded = []
        for detail in details:
            sigma = np.median(np.abs(detail)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(y)))
            thresholded.append(pywt.threshold(detail, threshold, mode=method))

        y_denoised = pywt.waverec([approximation, *thresholded], wavelet)
        df_denoised[col] = y_denoised[:len(y)]

    return df_denoised


def downsample_das(df, time_factor=4, space_factor=4, method='max_pool'):
    """Downsample in time and space with the chosen pooling method."""
    if method not in POOLING_METHODS:
        raise ValueError(f'Unknown pooling method {method!r}, expected one of {POOLING_METHODS}.')

    data = df.values
    nt, ns = data.shape

    nt_new = (nt // time_factor) * time_factor
    ns_new = (ns // space_factor) * space_factor
    data = data[:nt_new, :ns_new]

    reshaped = data.reshape(nt_new // time_factor, time_factor,
                            ns_new // space_factor, space_factor)

    if method == 'max_pool':
        downsampled = reshaped.max(axis=(1, 3))
    elif method == 'max_mean':
        downsampled = reshaped.max(axis=1).mean(axis=2)
    elif method == 'median':
        downsampled = np.median(reshaped, axis=(1, 3))
    elif method == 'median_max':
        downsampled = np.median(reshaped, axis=1).max(axis=2)
    else:  # local_max
        filtered = maximum_filter(data, size=(time_factor, space_factor))
        downsampled = filtered[::time_factor, ::space_factor]

    return pd.DataFrame(
        downsampled,
        index=df.index[:nt_new:time_factor],
        columns=df.columns[:ns_new:space_factor],
    )


def clip_percentile(df, upper_percentile=97):
    """Clip the top percentile to get rid of the extreme values."""
    upper = np.percentile(df.values, upper_percentile)
    return pd.DataFrame(np.clip(df.values, None, upper), index=df.index, columns=df.columns)


def bandpass_filter_das_time(df, min_freq=None, max_freq=None, dt=DT):
    """Zero out frequencies outside [min_freq, max_freq] along the time axis."""
    n_samples = len(df)
    fft_coeffs = np.fft.rfft(df.values, axis=0)
    freqs = np.fft.rfftfreq(n_samples, d=dt)

    mask = np.ones_like(freqs)
    if min_freq is not None:
        mask[freqs < min_freq] = 0.0
    if max_freq is not None:
        mask[freqs > max_freq] = 0.0

    filtered = np.fft.irfft(fft_coeffs * mask[:, np.newaxis], n=n_samples, axis=0)
    return pd.DataFrame(filtered, index=df.index, columns=df.columns)


def normalize_das_columns(df, target_sum=1.0):
    """Scale every channel to the same absolute sum, they have very different sensitivity."""
    data = df.values
    col_abs_sum = np.sum(np.abs(data), axis=0)
    col_abs_sum = np.where(col_abs_sum == 0, 1.0, col_abs_sum)
    return pd.DataFrame(data * (target_sum / col_abs_sum), index=df.index, columns=df.columns)


def spectrogram_simple(x, fs, nperseg=256, noverlap=128):
    """STFT with a Hann window, written by hand so it is visible in the report."""
    step = nperseg - noverlap
    n_segments = (len(x) - nperseg) // step + 1

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    times = np.arange(n_segments) * step / fs

    sxx = np.zeros((len(freqs), n_segments))
    for i in range(n_segments):
        start = i * step
        sxx[:, i] = np.abs(np.fft.rfft(x[start:start + nperseg] * window)) ** 2

    return freqs, times, sxx


def find_peaks_simple(data, height=None, distance=1):
    """Indices of local maxima, optionally filtered by height and min spacing."""
    peaks = [
        i for i in range(1, len(data) - 1)
        if data[i] > data[i - 1] and data[i] > data[i + 1]
        and (height is None or data[i] >= height)
    ]

    if distance > 1 and peaks:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= distance:
                filtered.append(p)
        peaks = filtered

    return np.array(peaks)
