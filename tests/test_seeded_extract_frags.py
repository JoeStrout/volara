import daisy
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import SeededExtractFrags
from volara.datasets import Affs, Labels
from volara.dbs import SQLite


def test_seeded_extract_frags_api_conformance():
    required = [
        "task_name",
        "write_roi",
        "write_size",
        "context_size",
        "drop_artifacts",
        "init",
        "process_block_func",
    ]
    for attr in required:
        assert hasattr(SeededExtractFrags, attr), f"Missing {attr}"


@pytest.mark.xfail(
    reason="SeededExtractFrags has hardcoded sigma=(0,6,9,9) for 4D data; "
    "does not work with 2D/3D affinities",
    raises=RuntimeError,
    strict=True,
)
def test_seeded_extract_frags_basic(tmp_path):
    """2D 10x10 affinities, 2 skeleton seeds, verify segments assigned to seeds.

    Note: SeededExtractFrags reads node_attrs["skeleton_id"] from graph nodes.
    The DB schema needs a 'skeleton_id' custom node_attr for this to work.

    This test is expected to fail because sigma=(0,6,9,9) is hardcoded for 4D.
    """
    # Create affs: strong affinities everywhere
    affs_data = np.ones((2, 10, 10), dtype=np.float32) * 0.9
    # Weaken affinities at the boundary (row 5) for offset [1,0]
    affs_data[0, 4, :] = 0.0

    affs_path = tmp_path / "data.zarr" / "affs"
    arr = prepare_ds(
        affs_path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = [[1, 0], [0, 1]]

    segs_path = tmp_path / "data.zarr" / "segs"

    db = SQLite(
        path=tmp_path / "data.zarr" / "skeletons.sqlite",
        node_attrs={"skeleton_id": "int"},
        ndim=2,
    )
    db.init()

    # Add skeleton seed nodes
    graph_provider = db.open("r+")
    g = graph_provider.read_graph()
    g.add_node(1, position=(2, 5), size=1, skeleton_id=100)
    g.add_node(2, position=(7, 5), size=1, skeleton_id=200)
    graph_provider.write_graph(g)

    task = SeededExtractFrags(
        affs_data=Affs(store=affs_path),
        segs_data=Labels(store=segs_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=[-0.5, -0.5],
        graph_db=db,
    )
    task.init()

    block = daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )

    with task.process_block_func() as process_block:
        process_block(block)

    segs = task.segs_data.array("r")[:]
    unique_labels = set(np.unique(segs)) - {0}
    assert len(unique_labels) >= 1, f"Expected segments, got {unique_labels}"
    assert 100 in unique_labels or 200 in unique_labels
