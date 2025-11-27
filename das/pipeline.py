"""The whole detection pipeline in one place, same steps as in the report."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .detection import binarize_per_channel, normalize_255, predict_lines, remove_specks, space_expand
from .signal import bandpass_filter_das_time, clip_percentile, denoise_das, downsample_das, normalize_das_columns


@dataclass
class PipelineConfig:
    """Parameters for detect_tracks. Defaults are the ones used in the report."""

    wavelet: str = 'sym8'
    wavelet_level: int = 6
    wavelet_method: str = 'garrote'
    max_freq: float = 30.0
    target_sum: float = 10.0
    upper_percentile: float = 97.0
    median_time_factor: int = 25
    local_max_time_factor: int = 4
    binarize_method: str = 'mad'
    binarize_threshold: float = 3.0
    min_speck_size: int = 30
    space_expand_factor: int = 10


@dataclass
class DetectionResult:
    """Detected lines plus all the intermediate frames so we can plot them."""

    lines: list = field(default_factory=list)
    detection_image: np.ndarray | None = None
    stages: dict[str, pd.DataFrame] = field(default_factory=dict)


def detect_tracks(df, config=None):
    """Run the whole pipeline on one DAS segment."""
    config = config or PipelineConfig()

    stages = {}
    stages['raw'] = df.abs()

    denoised = denoise_das(stages['raw'], wavelet=config.wavelet, level=config.wavelet_level,
                           method=config.wavelet_method).abs()
    stages['denoised'] = denoised

    stages['filtered'] = bandpass_filter_das_time(denoised, max_freq=config.max_freq)
    stages['normalized'] = normalize_das_columns(stages['filtered'], target_sum=config.target_sum)
    stages['clipped'] = clip_percentile(stages['normalized'], upper_percentile=config.upper_percentile)

    stages['median'] = downsample_das(stages['clipped'], time_factor=config.median_time_factor,
                                      space_factor=1, method='median')
    stages['local_max'] = downsample_das(stages['median'], time_factor=config.local_max_time_factor,
                                         space_factor=1, method='local_max')

    stages['binary'] = binarize_per_channel(stages['local_max'], method=config.binarize_method,
                                            threshold_val=config.binarize_threshold)

    cleaned = remove_specks(stages['binary'].to_numpy(), min_size=config.min_speck_size)
    detection_image = normalize_255(space_expand(cleaned, factor=config.space_expand_factor))

    return DetectionResult(
        lines=predict_lines(detection_image),
        detection_image=detection_image,
        stages=stages,
    )
