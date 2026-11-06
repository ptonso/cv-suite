"""Hash distances, components and SSIM: `common/similarity.py`."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cvsuite.common import similarity as sim


def noise(seed: int, size: int = 64) -> np.ndarray:
    return (np.random.default_rng(seed).random((size, size, 3)) * 255).astype(np.uint8)


def save(tmp_path: Path, name: str, array: np.ndarray) -> Path:
    path = tmp_path / name
    Image.fromarray(array).save(path)
    return path


def test_hash_bits_shape_and_identical_images(tmp_path: Path) -> None:
    paths = [save(tmp_path, "a.png", noise(0)), save(tmp_path, "b.png", noise(0)), save(tmp_path, "c.png", noise(1))]
    bits = sim.hash_bits(paths)
    assert bits.shape == (3, 8) and bits.dtype == np.uint8
    assert (bits[0] == bits[1]).all()
    assert sim.nearest_distance(bits[:1], bits[2:]).tolist() == [int(sim._block_distance(bits[:1], bits[2:])[0, 0])]


def test_unreadable_image_raises_and_names_the_path(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="bad.png"):
        sim.hash_bits([bad])


def test_distances_match_a_brute_force_count_for_any_chunk() -> None:
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 256, size=(50, 8), dtype=np.uint8)
    dist = np.array([[bin(int.from_bytes(a ^ b, "big")).count("1") for b in bits] for a in bits])
    expected = sorted((i, j) for i in range(50) for j in range(i + 1, 50) if dist[i, j] <= 26)
    assert expected
    for chunk in (1, 7, 1024):
        assert sim.pairs_within(bits, 26, chunk=chunk) == expected
    query, ref = bits[:20], bits[20:]
    nearest = dist[:20, 20:].min(axis=1)
    for chunk in (1, 6, 1024):
        assert sim.nearest_distance(query, ref, chunk=chunk).tolist() == nearest.tolist()


def test_components_use_the_smallest_member_index() -> None:
    assert sim.components(6, [(1, 2), (2, 4), (0, 5)]) == [0, 1, 1, 3, 1, 0]
    assert sim.components(3, []) == [0, 1, 2]


def test_thresholds_are_validated() -> None:
    bits = np.zeros((2, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        sim.pairs_within(bits, 65)
    with pytest.raises(ValueError):
        sim.check_ssim(1.5)
    with pytest.raises(ValueError):
        sim.nearest_distance(bits, bits[:0])


def test_ssim_is_one_for_equal_arrays_and_lower_for_noise() -> None:
    a = np.random.default_rng(0).random((40, 40)) * 255
    b = np.random.default_rng(1).random((40, 40)) * 255
    assert sim.ssim(a, a) == pytest.approx(1.0)
    assert sim.ssim(a, b) < 0.3
    with pytest.raises(ValueError):
        sim.ssim(a, b[:30])


def test_gray_thumb_has_a_fixed_shape(tmp_path: Path) -> None:
    path = save(tmp_path, "a.png", noise(0, 90))
    thumb = sim.gray_thumb(path, size=32)
    assert thumb.shape == (32, 32)


def test_nearest_matches_brute_force_for_any_chunk_and_breaks_ties_by_index() -> None:
    rng = np.random.default_rng(3)
    reference = rng.integers(0, 256, size=(37, 8), dtype=np.uint8)
    reference[5] = reference[20]
    query = np.vstack([reference[20], rng.integers(0, 256, size=(11, 8), dtype=np.uint8)])
    full = np.array([[bin(int.from_bytes(bytes(q ^ r), "big")).count("1") for r in reference] for q in query])
    for chunk in (1, 4, 1024):
        distance, index = sim.nearest(query, reference, 3, chunk=chunk)
        expected = np.argsort(full, axis=1, kind="stable")[:, :3]
        assert (index == expected).all()
        assert (distance == np.take_along_axis(full, expected, axis=1)).all()
    assert index[0, 0] == 5 and distance[0, 0] == 0
    with pytest.raises(ValueError):
        sim.nearest(query, reference[:0])
