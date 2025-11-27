"""Helper functions for the DAS project, pulled out of the notebook."""

from .config import DT, DX, MAX_VELOCITY_KMH, MIN_VELOCITY_KMH, SLOPE_SCALE
from .detection import (
    binarize_per_channel,
    check_intersection,
    get_parallel_line_distance,
    merge_lines,
    normalize_255,
    predict_lines,
    remove_specks,
    slope_to_velocity,
    space_expand,
)
from .io import load_segment
from .pipeline import DetectionResult, PipelineConfig, detect_tracks
from .plotting import (
    plot_das_data,
    plot_das_with_lines,
    plot_frequency_analysis,
    plot_spectrogram,
    set_axis,
    show_lines,
)
from .signal import (
    bandpass_filter_das_time,
    clip_percentile,
    denoise_das,
    downsample_das,
    find_peaks_simple,
    normalize_das_columns,
    spectrogram_simple,
)
