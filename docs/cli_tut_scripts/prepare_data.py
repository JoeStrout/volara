#!/usr/bin/env python
"""Prepare toy data for the Volara CLI tutorial.

Creates a zarr container (cells3d.zarr) with:
  - raw:    2-channel fluorescence microscopy (nuclei + membranes)
  - mask:   binary mask from simple thresholding
  - labels: pseudo ground truth via connected components
  - affs:   perfect affinities derived from labels

Usage:
    python prepare_data.py
"""

import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from scipy.ndimage import label
from skimage import data
from skimage.filters import gaussian

from volara.tmp import seg_to_affgraph

# Download the cells3d sample dataset and rearrange to (C, Z, Y, X)
cell_data = (data.cells3d().transpose((1, 0, 2, 3)) / 256).astype(np.uint8)

# Metadata
offset = Coordinate(0, 0, 0)
voxel_size = Coordinate(290, 260, 260)
axis_names = ["c^", "z", "y", "x"]
units = ["nm", "nm", "nm"]

# Raw data
print("Writing raw data...")
cell_array = prepare_ds(
    "cells3d.zarr/raw",
    cell_data.shape,
    offset=offset,
    voxel_size=voxel_size,
    axis_names=axis_names,
    units=units,
    mode="w",
    dtype=np.uint8,
)
cell_array[:] = cell_data

# Binary mask
print("Writing mask...")
mask_array = prepare_ds(
    "cells3d.zarr/mask",
    cell_data.shape[1:],
    offset=offset,
    voxel_size=voxel_size,
    axis_names=axis_names[1:],
    units=units,
    mode="w",
    dtype=np.uint8,
)
cell_mask = np.clip(gaussian(cell_data[1] / 255.0, sigma=1), 0, 255) * 255 > 30
not_membrane_mask = (
    np.clip(gaussian(cell_data[0] / 255.0, sigma=1), 0, 255) * 255 < 10
)
mask_array[:] = cell_mask * not_membrane_mask

# Labels via connected components
print("Writing labels...")
labels_array = prepare_ds(
    "cells3d.zarr/labels",
    cell_data.shape[1:],
    offset=offset,
    voxel_size=voxel_size,
    axis_names=axis_names[1:],
    units=units,
    mode="w",
    dtype=np.uint8,
)
labels_array[:] = label(mask_array[:])[0]

# Affinities from ground truth
print("Writing affinities...")
affs_array = prepare_ds(
    "cells3d.zarr/affs",
    (3,) + cell_data.shape[1:],
    offset=offset,
    voxel_size=voxel_size,
    axis_names=["offset^"] + axis_names[1:],
    units=units,
    mode="w",
    dtype=np.uint8,
)
affs_array[:] = (
    seg_to_affgraph(labels_array[:], nhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255
)

print("Done! Created cells3d.zarr with datasets: raw, mask, labels, affs")
