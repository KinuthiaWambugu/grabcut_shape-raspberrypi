"""Tests for core.image_io: resizing, preprocessing and 4-circle detection.

Synthetic images are generated in tmp_path, so these tests run on a clean
checkout (data/ is gitignored). The real-photo test is skipped when
data/raw/Scene4.jpeg is not present.
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # never open a window during tests
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.grabcut_shape_algorithm.core.image_io import (
    detect_four_circle_points,
    load_and_preprocess_image,
    load_detect_and_preprocess_image,
    resize_if_larger,
)

TARGET_SIZE = (480, 360)
REAL_IMAGE = Path("data/raw/Scene4.jpeg")

# (x, y) centres of the 4 dark circles in the synthetic image
SYNTH_CENTRES = [(120, 100), (680, 110), (130, 500), (670, 490)]


@pytest.fixture
def four_circles_path(tmp_path):
    """800x600 white image with 4 dark filled circles in the corners."""
    img = np.full((600, 800, 3), 255, np.uint8)
    for cx, cy in SYNTH_CENTRES:
        cv2.circle(img, (cx, cy), 40, (20, 20, 20), -1)
    path = tmp_path / "four_circles.png"
    cv2.imwrite(str(path), img)
    return path


def _save_visualization(image_path, bbox, centres, cropped_rgb, out_path):
    """Side-by-side figure: detection on the raw image / GMM input crop."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    raw = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    axes[0].imshow(raw)
    bx, by, bw, bh = bbox
    axes[0].add_patch(plt.Rectangle((bx, by), bw, bh, linewidth=2,
                                    edgecolor="lime", facecolor="none"))
    axes[0].scatter(centres[:, 0], centres[:, 1], c="red", s=80, marker="x")
    axes[0].set_title("Raw image detection")
    axes[1].imshow(cropped_rgb)
    axes[1].set_title(f"GMM input {cropped_rgb.shape[1]}x{cropped_rgb.shape[0]}")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_resize_if_larger_downscales_only_when_needed():
    big = np.zeros((1000, 1200, 3), np.uint8)
    small = np.zeros((200, 300, 3), np.uint8)
    assert resize_if_larger(big, TARGET_SIZE).shape[:2] == (TARGET_SIZE[1], TARGET_SIZE[0])
    assert resize_if_larger(small, TARGET_SIZE).shape == small.shape


def test_load_and_preprocess_returns_bgr_and_rgb(four_circles_path):
    resized_bgr, converted = load_and_preprocess_image(
        four_circles_path, target_size=TARGET_SIZE, color_space="rgb"
    )
    assert resized_bgr.shape == converted.shape
    assert resized_bgr.shape[1] <= TARGET_SIZE[0] and resized_bgr.shape[0] <= TARGET_SIZE[1]
    assert np.array_equal(converted, cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))


def test_load_and_preprocess_rejects_missing_file_and_bad_colorspace(four_circles_path, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_image(tmp_path / "nope.png")
    with pytest.raises(ValueError):
        load_and_preprocess_image(four_circles_path, color_space="lab")


def test_detect_four_circle_points_finds_the_four_circles(four_circles_path):
    img_rgb = cv2.cvtColor(cv2.imread(str(four_circles_path)), cv2.COLOR_BGR2RGB)
    bbox, centres = detect_four_circle_points(img_rgb)
    assert centres.shape == (4, 2)
    for ex, ey in SYNTH_CENTRES:
        dists = np.hypot(centres[:, 0] - ex, centres[:, 1] - ey)
        assert dists.min() < 8, f"no detected centre near {(ex, ey)}"
    x, y, w, h = bbox
    assert 0 <= x and 0 <= y and w > 0 and h > 0
    assert x + w <= img_rgb.shape[1] and y + h <= img_rgb.shape[0]


def test_detect_four_circle_points_raises_when_no_circles(tmp_path):
    blank = np.full((600, 800, 3), 255, np.uint8)
    with pytest.raises(ValueError, match="Could not detect 4 circles"):
        detect_four_circle_points(blank)


def test_load_detect_and_preprocess_crops_to_target(four_circles_path):
    cropped_rgb, bbox, centres = load_detect_and_preprocess_image(
        four_circles_path, target_size=TARGET_SIZE, color_space="rgb", detect_circles=True
    )
    assert len(centres) == 4
    assert cropped_rgb.shape[1] <= TARGET_SIZE[0] and cropped_rgb.shape[0] <= TARGET_SIZE[1]
    assert cropped_rgb.ndim == 3 and cropped_rgb.shape[2] == 3


@pytest.mark.skipif(not REAL_IMAGE.exists(), reason=f"{REAL_IMAGE} not available (data/ is gitignored)")
def test_real_scene_detection_and_visualization(tmp_path):
    cropped_rgb, bbox, centres = load_detect_and_preprocess_image(
        REAL_IMAGE, target_size=TARGET_SIZE, color_space="rgb", detect_circles=True
    )
    assert len(centres) == 4
    out = tmp_path / "test_detection_vis.png"
    _save_visualization(REAL_IMAGE, bbox, centres, cropped_rgb, out)
    assert out.is_file()