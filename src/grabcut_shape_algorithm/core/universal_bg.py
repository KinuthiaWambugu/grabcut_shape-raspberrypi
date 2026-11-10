from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

class UniversalBackgroundGMM:
    """Shared RGB background GMM trained on balanced samples from camera poses."""

    def __init__(
        self,
        n_components: int = 5,
        max_samples_per_pose: int = 25000,
        random_state: int = 42,
        reg_covar: float = 1e-5,
    ):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if max_samples_per_pose < n_components:
            raise ValueError("max_samples_per_pose must be >= n_components")
        if reg_covar <= 0:
            raise ValueError("reg_covar must be positive")

        self.n_components = n_components
        self.max_samples_per_pose = max_samples_per_pose
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.model: GaussianMixture | None = None
        self.pose_ids: tuple[str, ...] = ()
        self.training_samples_per_pose: dict[str, int] = {}

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name}: expected an HxWx3 colour image, got {image.shape}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"{name}: image is empty")
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError(f"{name}: image must contain numeric values")
        if not np.isfinite(image).all():
            raise ValueError(f"{name}: image contains NaN or infinity")
        return image.reshape(-1, 3).astype(np.float64, copy=False)

    def fit(
        self,
        background_images_by_pose: Mapping[str, Sequence[np.ndarray]],
    ) -> "UniversalBackgroundGMM":
        if not background_images_by_pose:
            raise ValueError("No background images supplied")

        pose_ids = sorted(background_images_by_pose)
        if any(not str(pose).strip() for pose in pose_ids):
            raise ValueError("Pose identifiers cannot be empty")

        pixels_by_pose = {}
        for pose_id in pose_ids:
            images = background_images_by_pose[pose_id]
            if not images:
                raise ValueError(f"{pose_id}: no images supplied")

            pixels = np.concatenate(
                [self._validate_image(image, pose_id) for image in images],
                axis=0,
            )
            if len(pixels) < self.n_components:
                raise ValueError(
                    f"{pose_id}: not enough pixels for {self.n_components} components"
                )
            pixels_by_pose[pose_id] = pixels

        samples_per_pose = min(
            self.max_samples_per_pose,
            min(len(pixels) for pixels in pixels_by_pose.values()),
        )
        rng = np.random.default_rng(self.random_state)
        balanced_samples = []

        for pose_id in pose_ids:
            pixels = pixels_by_pose[pose_id]
            indices = rng.choice(len(pixels), size=samples_per_pose, replace=False)
            balanced_samples.append(pixels[indices])

        training_pixels = np.concatenate(balanced_samples, axis=0)
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=100,
            n_init=3,
            reg_covar=self.reg_covar,
        )
        model.fit(training_pixels)

        self.model = model
        self.pose_ids = tuple(pose_ids)
        self.training_samples_per_pose = {
            pose_id: samples_per_pose for pose_id in pose_ids
        }
        return self

    def score_samples(
        self,
        image: np.ndarray,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Background GMM has not been fitted")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        image = np.asarray(image)
        pixels = self._validate_image(image, "input")
        h, w = image.shape[:2]
        scores = np.empty(len(pixels), dtype=np.float64)

        for start in range(0, len(pixels), chunk_size):
            end = min(start + chunk_size, len(pixels))
            scores[start:end] = self.model.score_samples(pixels[start:end])

        return scores.reshape(h, w)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted background GMM")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_components": self.n_components,
                "max_samples_per_pose": self.max_samples_per_pose,
                "random_state": self.random_state,
                "reg_covar": self.reg_covar,
                "pose_ids": self.pose_ids,
                "training_samples_per_pose": self.training_samples_per_pose,
                "color_space": "rgb",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "UniversalBackgroundGMM":
        payload = joblib.load(Path(path))
        if payload.get("color_space") != "rgb":
            raise ValueError("Unsupported saved model colour space")

        instance = cls(
            n_components=payload["n_components"],
            max_samples_per_pose=payload["max_samples_per_pose"],
            random_state=payload["random_state"],
            reg_covar=payload["reg_covar"],
        )
        instance.model = payload["model"]
        instance.pose_ids = tuple(payload["pose_ids"])
        instance.training_samples_per_pose = dict(
            payload["training_samples_per_pose"]
        )
        return instance
