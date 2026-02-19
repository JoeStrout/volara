import numpy as np

from volara.blockwise import ApplyShift, ComputeShift


def test_compute_shift_api_conformance():
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
        assert hasattr(ComputeShift, attr), f"Missing {attr}"


def test_apply_shift_api_conformance():
    required = [
        "task_name",
        "write_roi",
        "write_size",
        "context_size",
        "drop_artifacts",
        "process_block_func",
    ]
    for attr in required:
        assert hasattr(ApplyShift, attr), f"Missing {attr}"


def test_compute_shift_zero_shift():
    """Identical images produce ~0 shift for channel 0 (both channels use same target)."""
    # Create a structured 3D image (C=2, Z=9, Y=9, X=9) - divisible by 3 as required
    # Use a more structured image so cross-correlation works well
    base = np.zeros((9, 9, 9), dtype=np.float32)
    base[2:7, 2:7, 2:7] = 1.0  # clear feature in center
    image = np.stack([base, base], axis=0)  # both channels identical
    target = base.copy()
    voxel_size = np.array([1, 1, 1])

    shift = ComputeShift.compute_shift(image, target, voxel_size)
    assert shift.shape == (2, 3, 1, 1, 1)
    # At least channel 0 (identical to target) should have ~0 shift
    np.testing.assert_allclose(shift[0], 0, atol=1.0)
