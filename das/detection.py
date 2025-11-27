"""Binarisation, cleanup, Hough lines and merging them."""

import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks

from .config import DT, DX, MAX_VELOCITY_KMH, MIN_VELOCITY_KMH, SLOPE_SCALE

BINARIZE_METHODS = ('mad', 'std')


def binarize_per_channel(df, method='mad', threshold_val=3.0):
    """Threshold each channel separately, a global threshold kills the thin lines."""
    if method not in BINARIZE_METHODS:
        raise ValueError(f'Unknown method {method!r}, expected one of {BINARIZE_METHODS}.')

    abs_df = df.abs()

    if method == 'std':
        thresholds = abs_df.mean(axis=0) + threshold_val * abs_df.std(axis=0)
    else:
        medians = abs_df.median(axis=0)
        sigma_est = 1.4826 * (abs_df - medians).abs().median(axis=0)
        thresholds = medians + threshold_val * sigma_est

    return (abs_df > thresholds).astype(int)


def remove_specks(data, min_size=64):
    """Remove connected components smaller than min_size pixels."""
    labels = label(data.astype(bool), connectivity=2)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False  # label 0 is background
    return keep[labels].astype(data.dtype)


def space_expand(image, factor):
    """Repeat every column factor times, Hough is bad at angles on a narrow image."""
    return np.repeat(image, factor, axis=1)


def normalize_255(image):
    """Rescale to 0-255 uint8."""
    image = np.asarray(image, dtype=float)
    minimum, maximum = np.min(image), np.max(image)

    if np.isclose(maximum, minimum):
        return np.zeros(image.shape, dtype=np.uint8)

    return ((image - minimum) / (maximum - minimum) * 255).astype(np.uint8)


def slope_to_velocity(theta, dt=DT, dx=DX, slope_scale=SLOPE_SCALE):
    """Hough angle -> (slope, speed in km/h). See SLOPE_SCALE in config.py."""
    slope = np.tan(theta + np.pi / 2)
    denominator = slope * slope_scale * dt / dx

    if np.isclose(denominator, 0):
        return slope, 0.0

    return slope, abs(1 / denominator * 3.6)


def check_intersection(r1, t1, r2, t2, shape, margin=50):
    """True if the two Hough lines cross inside the image (+ a margin)."""
    matrix = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])

    if np.isclose(np.linalg.det(matrix), 0):
        return False

    try:
        x, y = np.linalg.solve(matrix, np.array([r1, r2]))
    except np.linalg.LinAlgError:
        return False

    height, width = shape[0], shape[1]
    return -margin <= x <= width + margin and -margin <= y <= height + margin


def get_parallel_line_distance(r1, t1, r2, t2, shape):
    """Distance between two almost parallel lines, measured at the image centre."""
    cx, cy = shape[1] / 2, shape[0] / 2
    d1 = r1 - (cx * np.cos(t1) + cy * np.sin(t1))
    d2 = r2 - (cx * np.cos(t2) + cy * np.sin(t2))
    return abs(d1 - d2)


def merge_lines(lines, img_shape, angle_thresh=0.15, dist_thresh=40):
    """Merge duplicate lines - skeletonisation gives a few of them per car.

    In: (rho, theta, slope, velocity, accumulator). Out: (rho, theta, slope, velocity).
    """
    if not lines:
        return []

    lines = sorted(lines, key=lambda item: item[4], reverse=True)
    clusters = []

    for line in lines:
        rho, theta, _, _, _ = line
        matched = False

        for cluster in clusters:
            ang_diff = abs(theta - cluster['avg_theta'])
            if ang_diff > np.pi:
                ang_diff = 2 * np.pi - ang_diff

            if ang_diff >= angle_thresh:
                continue

            intersects = check_intersection(rho, theta, cluster['avg_rho'],
                                            cluster['avg_theta'], img_shape)
            is_close = abs(rho - cluster['avg_rho']) < dist_thresh

            if intersects or is_close:
                cluster['lines'].append(line)
                weight = sum(item[4] for item in cluster['lines'])
                cluster['avg_rho'] = sum(item[0] * item[4] for item in cluster['lines']) / weight
                cluster['avg_theta'] = sum(item[1] * item[4] for item in cluster['lines']) / weight
                matched = True
                break

        if not matched:
            clusters.append({'lines': [line], 'avg_rho': rho, 'avg_theta': theta})

    results = []
    for cluster in clusters:
        slope, velocity = slope_to_velocity(cluster['avg_theta'])
        results.append((cluster['avg_rho'], cluster['avg_theta'], slope, velocity))

    return results


def predict_lines(img, angle_bins=360, min_velocity=MIN_VELOCITY_KMH,
                  max_velocity=MAX_VELOCITY_KMH, angle_thresh=0.15, dist_thresh=40):
    """Skeletonise, run Hough, drop the unrealistic speeds and merge the rest."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, angle_bins, endpoint=False)
    accumulator, angles, distances = hough_line(skeletonize(img > 0), theta=theta)

    raw_lines = []
    for acc, angle, dist in zip(*hough_line_peaks(accumulator, angles, distances), strict=True):
        slope, velocity = slope_to_velocity(angle)
        if min_velocity < velocity < max_velocity:
            raw_lines.append((dist, angle, slope, velocity, acc))

    return merge_lines(raw_lines, img.shape, angle_thresh=angle_thresh, dist_thresh=dist_thresh)
