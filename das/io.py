"""Loading the DAS .npy files into dataframes."""

import datetime
import os

import numpy as np
import pandas as pd

from .config import DT, DX, RECORDING_DATE


def load_segment(segment, files, dx=DX, dt=DT, date=RECORDING_DATE):
    """Load the .npy files inside a (start, end) HHMMSS window into one dataframe.

    Rows are timestamps, columns are the distance along the fibre in metres.
    """
    files = sorted(files)
    start_t = int(segment[0])
    end_t = int(segment[1])

    all_data = []
    all_times = []

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if not stem.isdigit():
            continue

        timestamp = int(stem)
        if not start_t <= timestamp <= end_t:
            continue

        arr = np.load(f)
        file_time = datetime.datetime.strptime(
            f'{date} {timestamp:06d}', '%Y-%m-%d %H%M%S'
        )
        all_data.append(arr)
        all_times.append(pd.date_range(start=file_time, periods=len(arr), freq=f'{dt}s'))

    if not all_data:
        # otherwise np.concatenate blows up with a confusing message
        raise ValueError(
            f'No files matched the window {segment[0]}-{segment[1]}. '
            f'Checked {len(files)} files, expected names like "092722.npy".'
        )

    data = np.concatenate(all_data)
    index = pd.DatetimeIndex(np.concatenate(all_times))
    columns = np.arange(data.shape[1]) * dx

    return pd.DataFrame(data=data, index=index, columns=columns)
