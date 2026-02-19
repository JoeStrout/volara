import numpy as np

from volara.lut import LUT, LUTS


def test_lut_save_load(tmp_path):
    lut = LUT(path=tmp_path / "test_lut.npz")
    data = np.array([[1, 2, 3], [10, 20, 30]])
    lut.save(data)
    loaded = lut.load()
    assert loaded is not None
    np.testing.assert_array_equal(loaded, data)


def test_lut_load_missing(tmp_path):
    lut = LUT(path=tmp_path / "nonexistent.npz")
    assert lut.load() is None


def test_lut_drop(tmp_path):
    lut = LUT(path=tmp_path / "drop_test.npz")
    lut.save(np.array([[1], [2]]))
    assert lut.file.exists()
    lut.drop()
    assert not lut.file.exists()


def test_lut_path_extension(tmp_path):
    """`.npz` appended when missing from string path."""
    lut = LUT(path=str(tmp_path / "no_ext"))
    assert lut.file.suffix == ".npz"
    assert str(lut.file).endswith("no_ext.npz")

    # If already has .npz, don't double it
    lut2 = LUT(path=str(tmp_path / "has_ext.npz"))
    assert str(lut2.file).endswith("has_ext.npz")


def test_lut_add_creates_luts(tmp_path):
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "b.npz")
    result = lut_a + lut_b
    assert isinstance(result, LUTS)
    assert len(result.luts) == 2


def test_luts_load(tmp_path):
    """Concatenation of multiple LUTs."""
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "b.npz")
    lut_a.save(np.array([[1, 2], [10, 20]]))
    lut_b.save(np.array([[3, 4], [30, 40]]))
    luts = LUTS(luts=[lut_a, lut_b])
    loaded = luts.load()
    assert loaded.shape == (2, 4)
    np.testing.assert_array_equal(loaded, [[1, 2, 3, 4], [10, 20, 30, 40]])


def test_luts_load_iterated(tmp_path):
    """Chained lookup tables: frag->mid->seg."""
    lut_a = LUT(path=tmp_path / "a.npz")
    lut_b = LUT(path=tmp_path / "b.npz")
    # a: 1->10, 2->20
    lut_a.save(np.array([[1, 2], [10, 20]]))
    # b: 10->100, 20->200
    lut_b.save(np.array([[10, 20], [100, 200]]))
    luts = LUTS(luts=[lut_a, lut_b])
    loaded = luts.load_iterated()
    # Should chain: 1->100, 2->200
    np.testing.assert_array_equal(loaded[0], [1, 2])
    np.testing.assert_array_equal(loaded[1], [100, 200])
