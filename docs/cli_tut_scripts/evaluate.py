#!/usr/bin/env python
"""Evaluate segmentation accuracy against pseudo ground truth.

Compares segments to labels at a given Z-slice and reports false merges,
false splits, and overall accuracy.

Usage:
    python evaluate.py                          # default z=30
    python evaluate.py -z 15                    # choose slice
"""

import argparse

import numpy as np
import zarr


def evaluate(z: int = 30) -> None:
    segments = np.array(zarr.open("cells3d.zarr/segments", mode="r")[z])
    labels = np.array(zarr.open("cells3d.zarr/labels", mode="r")[z])

    s_to_l: dict[int, int] = {}
    false_merges = 0
    l_to_s: dict[int, int] = {}
    false_splits = 0

    for s, l in zip(segments.flat, labels.flat):
        if s not in s_to_l:
            s_to_l[s] = l
        elif s_to_l[s] != l:
            false_merges += 1
            print(f"Falsely merged labels: ({l}, {s_to_l[s]}) with segment {s}")
        if l not in l_to_s:
            l_to_s[l] = s
        elif l_to_s[l] != s:
            false_splits += 1
            print(f"Falsely split label: {l} into segments ({s}, {l_to_s[l]})")

    print(f"False merges:  {false_merges}")
    print(f"False splits:  {false_splits}")
    accuracy = (len(s_to_l) - (false_merges + false_splits)) / len(s_to_l)
    print(f"Accuracy:      {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation accuracy.")
    parser.add_argument("-z", type=int, default=30, help="Z-slice index (default: 30)")
    args = parser.parse_args()
    evaluate(args.z)
