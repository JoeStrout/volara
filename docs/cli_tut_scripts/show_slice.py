#!/usr/bin/env python
"""Save a 2D slice of a zarr array as a PNG image.

Usage:
    python show_slice.py cells3d.zarr/raw              # default z=30, output=slice.png
    python show_slice.py cells3d.zarr/raw -z 15        # choose slice
    python show_slice.py cells3d.zarr/raw -o raw.png   # choose output filename
    python show_slice.py cells3d.zarr/fragments -z 30 -o frags.png
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import zarr


def save_slice(store_path: str, z: int = 30, output: str = "slice.png") -> None:
    arr = zarr.open(store_path, mode="r")
    if not isinstance(arr, zarr.Array):
        raise SystemExit(f"Expected a zarr array at '{store_path}', got a group.")

    # Determine the slice based on array shape.
    # Shapes we expect:
    #   (Z, Y, X)         -> labels, fragments, segments
    #   (C, Z, Y, X)      -> raw (2-ch), affinities (3-ch)
    ndim = arr.ndim
    if ndim == 3:
        data = np.array(arr[z])
    elif ndim == 4:
        data = np.array(arr[:, z])
    else:
        raise SystemExit(f"Unsupported array dimensions: {ndim} (expected 3 or 4)")

    # Render based on data type and shape.
    if data.dtype in (np.uint32, np.uint64):
        # Label data: randomize non-zero label colors for visibility.
        data = data.astype(np.float32)
        labels = [x for x in np.unique(data) if x != 0]
        relabelling = random.sample(range(1, len(labels) + 1), len(labels))
        for old, new in zip(labels, relabelling):
            data[data == old] = new
        plt.imsave(output, data, cmap="jet")
    elif data.ndim == 2:
        # Single-channel grayscale.
        plt.imsave(output, data, cmap="gray")
    elif data.shape[0] == 2:
        # 2-channel: map to (ch0, ch1, ch0) -> magenta/green composite.
        rgb = np.stack([data[0], data[1], data[0]], axis=-1)
        plt.imsave(output, rgb)
    elif data.shape[0] == 3:
        # 3-channel (e.g. affinities): treat as RGB.
        plt.imsave(output, np.moveaxis(data, 0, -1))
    else:
        raise SystemExit(f"Don't know how to display {data.shape[0]}-channel data")

    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a zarr slice as PNG.")
    parser.add_argument("store", help="Path to zarr array (e.g. cells3d.zarr/raw)")
    parser.add_argument("-z", type=int, default=30, help="Z-slice index (default: 30)")
    parser.add_argument("-o", "--output", default="slice.png", help="Output PNG path")
    args = parser.parse_args()
    save_slice(args.store, args.z, args.output)
