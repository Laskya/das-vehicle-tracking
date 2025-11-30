# DAS Vehicle Tracking

CV project 1. Detecting cars in Distributed Acoustic Sensing data. A car driving
next to the fibre shows up as a diagonal line in the space-time image, and the
slope of the line tells us how fast it was going.

Jakub Laskowski (160287), Jakub Górniak (160326)

## Pipeline

wavelet denoising -> low pass filter -> normalisation per channel -> clipping ->
downsampling -> binarisation per channel -> removing small components ->
skeletonisation -> Hough transform -> merging the lines

## How to run

```bash
pip install -r requirements.txt
jupyter notebook cv_project1_final_improved.ipynb
```

The first cell downloads the data from Google Drive. All the helper functions are
in the `das` package.
