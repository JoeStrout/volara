from pathlib import Path

import daisy
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.dbs import SQLite
from volara.logging import set_log_basedir
from volara.tmp import seg_to_affgraph


@pytest.fixture(autouse=True)
def logdir(tmp_path):
    set_log_basedir(tmp_path / "volara_logs")


@pytest.fixture()
def zarr_2d(tmp_path) -> tuple[Path, np.ndarray]:
    """10x10 float32 array, voxel_size=(1,1), values in [0,1]."""
    data = np.random.default_rng(42).random((10, 10), dtype=np.float32)
    path = tmp_path / "test.zarr" / "raw"
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    return path, data


@pytest.fixture()
def labels_2d(tmp_path) -> tuple[Path, np.ndarray]:
    """10x10 uint64 with 4 labeled regions (2x2 grid of 5x5 blocks)."""
    data = np.zeros((10, 10), dtype=np.uint64)
    data[:5, :5] = 1
    data[:5, 5:] = 2
    data[5:, :5] = 3
    data[5:, 5:] = 4
    path = tmp_path / "test.zarr" / "labels"
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    return path, data


@pytest.fixture()
def affs_2d(tmp_path, labels_2d) -> tuple[Path, np.ndarray]:
    """(2,10,10) affinities from labels_2d using seg_to_affgraph."""
    labels_path, labels_data = labels_2d
    nhood = [[1, 0], [0, 1]]
    affs_data = seg_to_affgraph(labels_data, nhood=nhood).astype(np.float32)
    path = tmp_path / "test.zarr" / "affs"
    arr = prepare_ds(
        path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = nhood
    return path, affs_data


@pytest.fixture()
def frags_2d(tmp_path) -> tuple[Path, np.ndarray]:
    """10x10 uint64 with 10 horizontal stripe fragments (1..10)."""
    data = np.zeros((10, 10), dtype=np.uint64)
    data[:, :] = np.arange(1, 11)[:, None]
    path = tmp_path / "test.zarr" / "frags"
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    return path, data


@pytest.fixture()
def sqlite_db_2d(tmp_path) -> SQLite:
    """SQLite DB with ndim=2, ready for use."""
    db_config = SQLite(
        path=tmp_path / "test.zarr" / "db.sqlite",
        node_attrs={"raw_intensity": 1},
        edge_attrs={"y_aff": "float"},
        ndim=2,
    )
    db_config.init()
    return db_config


@pytest.fixture()
def block_2d() -> daisy.Block:
    """daisy.Block covering Roi((0,0),(10,10))."""
    return daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )
