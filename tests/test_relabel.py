import numpy as np
from funlib.geometry import Coordinate

from volara.blockwise import Relabel
from volara.datasets import Labels
from volara.lut import LUT


def test_relabel_init_and_drop(labels_2d, tmp_path):
    """init() creates output zarr, drop_artifacts() removes it."""
    frags_path, _ = labels_2d
    seg_path = tmp_path / "test.zarr" / "seg"
    lut = LUT(path=tmp_path / "lut.npz")
    lut.save(np.array([[1, 2, 3, 4], [10, 20, 30, 40]]))

    task = Relabel(
        frags_data=Labels(store=frags_path),
        seg_data=Labels(store=seg_path),
        lut=lut,
        block_size=Coordinate(10, 10),
    )
    task.init()
    assert seg_path.exists()
    task.drop_artifacts()
    assert not seg_path.exists()


def test_relabel_basic(labels_2d, block_2d, tmp_path):
    """Fragments [1,2,3,4] mapped to segments [10,20,30,40] via LUT."""
    frags_path, frags_data = labels_2d
    seg_path = tmp_path / "test.zarr" / "seg"
    lut = LUT(path=tmp_path / "lut.npz")
    lut.save(np.array([[1, 2, 3, 4], [10, 20, 30, 40]]))

    task = Relabel(
        frags_data=Labels(store=frags_path),
        seg_data=Labels(store=seg_path),
        lut=lut,
        block_size=Coordinate(10, 10),
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = task.seg_data.array("r")[:]
    expected = np.zeros_like(frags_data, dtype=np.uint64)
    for frag_id, seg_id in [(1, 10), (2, 20), (3, 30), (4, 40)]:
        expected[frags_data == frag_id] = seg_id
    np.testing.assert_array_equal(result, expected)
