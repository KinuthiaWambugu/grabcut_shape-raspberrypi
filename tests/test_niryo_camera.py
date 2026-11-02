import cv2
import numpy as np
import pytest

pytest.importorskip("pyniryo")

from src.grabcut_shape_algorithm.camera.niryo_camera import (
    capture_niryo_images,
)
from src.grabcut_shape_algorithm.config import DEFAULT_CONFIG as cfg


def _check_image(path):
    img = cv2.imread(str(path))
    assert img is not None, f"{path} is not a readable image"
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    assert img.std() > 0, f"{path} is a flat image (camera returned no content)"


def test_capture_observation_pose(tmp_path):
    """Capture only the observation pose and check the saved image + intrinsics."""
    label = cfg.observation_pose
    saved = capture_niryo_images(
        cfg.robot_ip,
        {label: cfg.niryo_poses[label]},
        tmp_path,
        capture_home=False,
    )

    assert label in saved
    assert saved[label] == tmp_path / f"{label}.png"
    _check_image(saved[label])

    intrinsics = np.load(tmp_path / "camera_intrinsics.npz")
    assert intrinsics["camera_matrix"].shape == (3, 3)
    assert intrinsics["distortion_coefficients"].size >= 4


def test_capture_all_poses_and_home(tmp_path):
    """Full capture sequence: every configured pose plus HOME."""
    saved = capture_niryo_images(cfg.robot_ip, cfg.niryo_poses, tmp_path)

    expected = set(cfg.niryo_poses) | {"HOME"}
    assert set(saved) == expected
    for path in saved.values():
        _check_image(path)