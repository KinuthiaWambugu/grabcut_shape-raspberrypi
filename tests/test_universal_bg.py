
import numpy as np
import pytest

from src.grabcut_shape_algorithm.core.gmm_priors import DualGMM


def make_background_images():
    rng = np.random.default_rng(42)

    return [
        np.clip(
            rng.normal(
                loc=(180 + i * 5, 185 + i * 3, 190 + i * 2),
                scale=8,
                size=(60, 60, 3),
            ),
            0,
            255,
        ).astype(np.uint8)
        for i in range(5)
    ]


def test_fit_bg_from_multiple_images():
    images = make_background_images()

    analyzer = DualGMM(
        k_bg=5,
        k_scene=12,
        max_samples=5000,
        random_state=42,
    )

    result = analyzer.fit_bg_from_images(images)

    assert result is analyzer
    assert analyzer.gmm_bg is not None
    assert analyzer.gmm_bg.means_.shape == (5, 3)
    assert analyzer.gmm_bg.weights_.shape == (5,)
    assert np.isclose(analyzer.gmm_bg.weights_.sum(), 1.0)


def test_universal_background_model_can_score_pixels():
    images = make_background_images()

    analyzer = DualGMM(k_bg=5, max_samples=5000, random_state=42)
    analyzer.fit_bg_from_images(images)

    test_pixels = images[0][:10, :10].reshape(-1, 3).astype(np.float64)
    scores = analyzer.gmm_bg.score_samples(test_pixels)

    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_fit_bg_from_images_rejects_empty_list():
    analyzer = DualGMM()

    with pytest.raises(ValueError):
        analyzer.fit_bg_from_images([])


def test_fit_bg_from_images_rejects_wrong_image_shape():
    analyzer = DualGMM()

    with pytest.raises(ValueError):
        analyzer.fit_bg_from_images([
            np.zeros((50, 50), dtype=np.uint8)
        ])


def test_fit_bg_from_images_rejects_non_finite_pixels():
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan

    analyzer = DualGMM()

    with pytest.raises(ValueError):
        analyzer.fit_bg_from_images([image])
