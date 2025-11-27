"""Constants for the DAS recording from 2024-05-07."""

# channel spacing along the fibre [m]
DX = 5.106500953873407

# sampling interval [s], so 625 Hz
DT = 0.0016

# the file names only have HHMMSS, so we need the date from somewhere
RECORDING_DATE = '2024-05-07'

# Scaling between the raw sample coordinates and the image the Hough transform
# sees. We downsample time by 25 and then by 4 and stretch space by 10, so it
# should be 1000, but 1100 matched the speeds we measured on the road better.
SLOPE_SCALE = 1100.0

# anything outside this speed range is treated as noise, not a car [km/h]
MIN_VELOCITY_KMH = 30.0
MAX_VELOCITY_KMH = 140.0
