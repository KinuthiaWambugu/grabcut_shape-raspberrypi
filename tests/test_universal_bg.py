import numpy as np
import pytest

from src.grabcut_shape_algorithm.core.universal_bg import UniversalBackgroundGMM


@pytest.fixture
def background_images():
    rng = np.random.default_rng(10)
    p1 = np.clip(rng.normal(195, 8, (100, 100, 3)), 0, 255).astype(np.uint8)
    p2 = np.clip(rng.normal(189, 8, (100, 100, 3)), 0, 255).astype(np.uint8)
    p6 = np.clip(rng.normal(196, 8, (100, 100, 3)), 0, 255).astype(np.uint8)
    return {"P1": [p1], "P2": [p2], "P6": [p6]}


def make_model(background_images):
    return UniversalBackgroundGMM(
        n_components=5, max_samples_per_pose=500
    ).fit(background_images)


def test_fit_universal_bg_from_multiple_poses(background_images):
    model = make_model(background_images)
    assert model.model is not None
    assert model.model.n_components == 5
    assert model.pose_ids == ("P1", "P2", "P6")
    assert set(model.training_samples_per_pose.values()) == {500}


def test_score_samples_returns_image_sized_map(background_images):
    model = make_model(background_images)
    image = background_images["P1"][0]
    scores = model.score_samples(image, chunk_size=1000)
    assert scores.shape == image.shape[:2]
    assert np.isfinite(scores).all()


def test_chunked_scoring_matches_large_chunk(background_images):
    model = make_model(background_images)
    image = background_images["P2"][0]
    small = model.score_samples(image, chunk_size=137)
    large = model.score_samples(image, chunk_size=100000)
    np.testing.assert_allclose(small, large, rtol=1e-10, atol=1e-10)


def test_save_and_load_preserves_scores(background_images, tmp_path):
    model = make_model(background_images)
    image = background_images["P6"][0]
    original = model.score_samples(image)
    path = tmp_path / "universal_bg.joblib"
    model.save(path)
    loaded = UniversalBackgroundGMM.load(path)
    assert loaded.pose_ids == model.pose_ids
    assert loaded.training_samples_per_pose == model.training_samples_per_pose
    np.testing.assert_allclose(original, loaded.score_samples(image), rtol=1e-10, atol=1e-10)


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError, match="No background images"):
        UniversalBackgroundGMM().fit({})


def test_fit_rejects_empty_pose():
    with pytest.raises(ValueError, match="no images supplied"):
        UniversalBackgroundGMM().fit({"P1": []})


def test_fit_rejects_invalid_image_shape():
    image = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        UniversalBackgroundGMM().fit({"P1": [image]})


def test_score_rejects_unfitted_model():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not been fitted"):
        UniversalBackgroundGMM().score_samples(image)


def test_save_rejects_unfitted_model(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        UniversalBackgroundGMM().save(tmp_path / "model.joblib")


def test_nonfinite_image_values_are_rejected(background_images):
    model = make_model(background_images)
    invalid = np.full((20, 20, 3), np.nan)
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.score_samples(invalid)


def test_invalid_chunk_size_is_rejected(background_images):
    model = make_model(background_images)
    with pytest.raises(ValueError, match="chunk_size"):
        model.score_samples(background_images["P1"][0], chunk_size=0)
