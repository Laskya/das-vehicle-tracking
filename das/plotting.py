"""Plots for the DAS frames, spectra and the detected lines."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, PowerNorm

from .config import DT
from .signal import spectrogram_simple


def set_axis(x, no_labels=7):
    """Evenly spaced tick positions + their labels."""
    n = x.shape[0]
    step = max(int(n / (no_labels - 1)), 1)
    return np.arange(0, n, step), x[::step]


def _build_norm(values, percentiles=(3, 99), norm_type='linear'):
    """Matplotlib norm built from percentiles, so both frame plots look the same."""
    low, high = np.percentile(values, percentiles)

    if norm_type == 'log':
        return LogNorm(vmin=max(low, 1e-10), vmax=max(high, 1e-10), clip=True)
    if norm_type == 'power':
        return PowerNorm(gamma=0.5, vmin=low, vmax=high, clip=True)
    return Normalize(vmin=low, vmax=high, clip=True)


def _decorate(fig, ax, im, df, title, no_labels):
    """Labels, colour bar and the time/space ticks - shared by both frame plots."""
    ax.set_ylabel('time')
    ax.set_xlabel('space [m]')
    ax.set_title(title)

    cax = fig.add_axes([
        ax.get_position().x1 + 0.06,
        ax.get_position().y0,
        0.02,
        ax.get_position().height,
    ])
    fig.colorbar(im, cax=cax)

    x_positions, x_labels = set_axis(df.columns, no_labels)
    ax.set_xticks(x_positions, np.round(x_labels))

    y_positions, y_labels = set_axis(df.index.time, no_labels)
    ax.set_yticks(y_positions, y_labels)


def plot_das_data(data, title='DAS Data', figsize=(12, 16), percentiles=(3, 99),
                  colormap='seismic', norm_type='linear', no_labels=7):
    """Waterfall plot of a space-time frame."""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    im = ax.imshow(data, interpolation='none', aspect='auto',
                   norm=_build_norm(data, percentiles, norm_type), cmap=colormap)

    _decorate(fig, ax, im, data, title, no_labels)
    return fig


def plot_das_with_lines(data, lines, processed_shape=(750, 520),
                        title='DAS Data with Detected Lines', figsize=(12, 16),
                        percentiles=(3, 99), colormap='seismic',
                        norm_type='linear', no_labels=7):
    """Waterfall plot with the detected lines and their speeds drawn on top.

    processed_shape is the shape of the downsampled image the lines come from.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    orig_h, orig_w = data.shape
    proc_h, proc_w = processed_shape
    scale_x, scale_y = orig_w / proc_w, orig_h / proc_h

    im = ax.imshow(data, interpolation='none', aspect='auto',
                   norm=_build_norm(data, percentiles, norm_type), cmap=colormap)

    for dist, angle, slope, velocity in lines:
        x0 = dist * np.cos(angle) * scale_x
        y0 = dist * np.sin(angle) * scale_y
        scaled_slope = slope * (scale_y / scale_x)

        ax.axline((x0, y0), slope=scaled_slope, color='lime', lw=2, ls='--', alpha=0.8)

        tx = orig_w * 0.45
        ty = scaled_slope * (tx - x0) + y0
        if not 0 <= ty <= orig_h:
            tx, ty = x0, y0

        ax.text(tx, ty, f'{velocity:.0f} km/h', color='white', fontsize=10,
                fontweight='bold', ha='center', va='bottom',
                bbox={'facecolor': 'green', 'alpha': 0.7, 'edgecolor': 'none',
                      'boxstyle': 'round,pad=0.2'})

    ax.set_xlim(0, orig_w)
    ax.set_ylim(orig_h, 0)

    _decorate(fig, ax, im, data, title, no_labels)
    return fig


def show_lines(img, lines, figsize=(12, 16)):
    """Draw the detected lines straight on the binary detection image."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray')

    height, width = img.shape[:2]
    ax.set_ylim((height, 0))
    ax.set_xlim((0, width))

    for dist, angle, slope, velocity in lines:
        x0 = dist * np.cos(angle)
        y0 = dist * np.sin(angle)

        ax.axline((x0, y0), slope=slope, color='red', linewidth=2, linestyle='--')

        tx = width / 2
        ty = slope * (width / 2 - x0) + y0
        if ty < 0 or ty > height:
            tx, ty = x0, y0

        ax.text(tx, ty, f'{velocity:.0f} km/h', color='white', fontsize=10,
                fontweight='bold', ha='center', va='bottom',
                bbox={'facecolor': 'red', 'edgecolor': 'none', 'boxstyle': 'round,pad=0.2'})

    fig.tight_layout()
    return fig


def plot_spectrogram(data, channel=None, dt=DT, nperseg=256, noverlap=128,
                     freq_limit=(0, 100), figsize=(12, 6), colormap='viridis'):
    """Spectrogram of a single channel."""
    if channel is None:
        channel = len(data.columns) // 2

    col = data.columns[channel] if isinstance(channel, int) else channel
    channel_label = f'Channel {channel}' if isinstance(channel, int) else str(channel)

    freqs, times, sxx = spectrogram_simple(data[col].values, fs=1.0 / dt,
                                           nperseg=nperseg, noverlap=noverlap)
    sxx_db = 10 * np.log10(sxx + 1e-10)

    fig, ax = plt.subplots(figsize=figsize)
    norm = Normalize(vmin=np.percentile(sxx_db, 3), vmax=np.percentile(sxx_db, 99))
    im = ax.pcolormesh(times, freqs, sxx_db, cmap=colormap, norm=norm, rasterized=True)

    ax.set_ylim(freq_limit)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_title(f'Spectrogram - {channel_label}')
    ax.tick_params(axis='both', which='major', labelsize=10)

    fig.colorbar(im, ax=ax, label='Power [dB]', pad=0.02).ax.tick_params(labelsize=10)
    return fig


def plot_frequency_analysis(data, dt=DT, title_prefix=''):
    """Per channel spectra, the average spectrum and the power per channel."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    n_samples = len(data)
    freqs = np.fft.fftfreq(n_samples, dt)[:n_samples // 2]

    ax1 = axes[0]
    for ch_idx in [0, 13, 26, 39, 51]:
        if ch_idx < len(data.columns):
            col = data.columns[ch_idx]
            spectrum = np.abs(np.fft.fft(data[col].values))[:n_samples // 2]
            ax1.plot(freqs, 20 * np.log10(spectrum + 1e-10), label=f'Ch {ch_idx} ({col})', alpha=0.7)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title(f'{title_prefix}Frequency Spectra (Selected Channels)')
    ax1.legend()
    ax1.set_xlim([0, 100])
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    avg_spectrum = np.zeros(n_samples // 2)
    for col in data.columns:
        avg_spectrum += np.abs(np.fft.fft(data[col].values))[:n_samples // 2]
    avg_spectrum /= len(data.columns)
    ax2.plot(freqs, 20 * np.log10(avg_spectrum + 1e-10), 'b-', linewidth=1.5)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude [dB]')
    ax2.set_title(f'{title_prefix}Average Spectrum Across All Channels')
    ax2.set_xlim([0, 150])
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    channel_power = [np.var(data[col]) for col in data.columns]
    ax3.bar(range(len(channel_power)), channel_power, color='steelblue')
    ax3.set_xlabel('Channel Index')
    ax3.set_ylabel('Signal Variance')
    ax3.set_title(f'{title_prefix}Signal Power Distribution Across Channels')
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
