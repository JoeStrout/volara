import numpy as np
from funlib.geometry import Coordinate

from volara.blockwise import ExtractFrags
from volara.datasets import Affs, Labels
from volara.dbs import SQLite


def test_extract_frags_init_and_drop(affs_2d, tmp_path):
    """init() creates both frags zarr and DB; drop_artifacts() removes them."""
    affs_path, _ = affs_2d
    frags_path = tmp_path / "test.zarr" / "frags"
    db = SQLite(path=tmp_path / "test.zarr" / "ef_db.sqlite", ndim=2)

    task = ExtractFrags(
        db=db,
        affs_data=Affs(store=affs_path),
        frags_data=Labels(store=frags_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=[-0.5, -0.5],
    )
    task.init()
    assert frags_path.exists()
    assert db.path.exists()

    task.drop_artifacts()
    assert not frags_path.exists()
    assert not db.path.exists()


def test_extract_frags_basic(affs_2d, block_2d, tmp_path):
    """Affinities from 4-quadrant labels produce fragments and DB nodes.

    affs_2d is derived from labels_2d (4 regions in a 2x2 grid).
    Affinities are 0 at region boundaries and 1 within regions,
    so watershed should produce at least 2 distinct fragments.
    """
    affs_path, _ = affs_2d
    frags_path = tmp_path / "test.zarr" / "frags"
    db = SQLite(path=tmp_path / "test.zarr" / "ef_db.sqlite", ndim=2)

    task = ExtractFrags(
        db=db,
        affs_data=Affs(store=affs_path),
        frags_data=Labels(store=frags_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=[-0.5, -0.5],
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    frags = task.frags_data.array("r")[:]
    unique_ids = set(np.unique(frags)) - {0}
    assert len(unique_ids) >= 2, f"Expected >=2 fragments, got {unique_ids}"

    # Verify nodes were added to the DB with positions
    graph = db.open("r").read_graph()
    assert graph.number_of_nodes() >= 2
    for _, attrs in graph.nodes(data=True):
        assert "position" in attrs
        assert "size" in attrs
