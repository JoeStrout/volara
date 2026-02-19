import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import AffAgglom
from volara.datasets import Affs, Labels


def test_aff_agglom_drop_edges(frags_2d, sqlite_db_2d, block_2d, tmp_path):
    """drop_artifacts() removes edges but preserves nodes."""
    frags_path, _ = frags_2d

    # Create 1-channel affs (needed for the task config, not for the drop test)
    affs_data = np.zeros((1, 10, 10), dtype=np.float32)
    affs_path = tmp_path / "test.zarr" / "affs"
    arr = prepare_ds(
        affs_path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = [[1, 0]]

    # Seed the DB with nodes and an edge
    db = sqlite_db_2d.open("r+")
    g = db.read_graph()
    g.add_node(1, position=(1, 5), size=1, raw_intensity=(1,))
    g.add_node(2, position=(2, 5), size=1, raw_intensity=(2,))
    g.add_edge(1, 2, y_aff=0.5)
    db.write_graph(g)

    # Verify edges exist before drop
    g = sqlite_db_2d.open("r").read_graph()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1

    task = AffAgglom(
        db=sqlite_db_2d,
        frags_data=Labels(store=frags_path),
        affs_data=Affs(store=affs_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores={"y_aff": [Coordinate(1, 0)]},
    )
    task.drop_artifacts()

    # Edges should be gone, nodes should remain
    g = sqlite_db_2d.open("r").read_graph()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0


def test_aff_agglom_basic(frags_2d, sqlite_db_2d, block_2d, tmp_path):
    """10 horizontal stripe fragments, alternating affinities -> 9 edges with correct scores."""
    frags_path, _ = frags_2d

    # Seed DB with 10 fragment nodes matching the stripe labels (1..10)
    db = sqlite_db_2d.open("r+")
    g = db.read_graph()
    for i in range(1, 11):
        g.add_node(i, position=(i, 5), size=1, raw_intensity=(i,))
    db.write_graph(g)

    # Create 1-channel affs with offset [1,0]: 1 in every other row, 0 elsewhere
    affs_data = np.zeros((1, 10, 10), dtype=np.uint32)
    affs_data[0, ::2, :] = 1
    affs_path = tmp_path / "test.zarr" / "affs"
    arr = prepare_ds(
        affs_path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = [[1, 0]]

    task = AffAgglom(
        db=sqlite_db_2d,
        frags_data=Labels(store=frags_path),
        affs_data=Affs(store=affs_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores={"y_aff": [Coordinate(1, 0)]},
    )

    with task.process_block_func() as process_block:
        process_block(block_2d)

    g = sqlite_db_2d.open("r").read_graph(block_2d.write_roi)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 9
    for u, v, data in g.edges(data=True):
        if u % 2 == 0:
            assert data["y_aff"] == 0.0
        else:
            assert data["y_aff"] == 1.0
