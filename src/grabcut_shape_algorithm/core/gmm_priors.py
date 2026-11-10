import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Any, Dict, Optional
import cv2

class DualGMM:
    def __init__(
        self,
        k_bg:int = 5,
        k_scene: int = 12,
        max_samples: int = 25000,
        random_state: int = 0,
    ):
        self.k_bg = k_bg
        self.k_scene = k_scene
        self.max_samples = max_samples
        self.random_state = random_state

        self.gmm_bg: GaussianMixture | None = None
        self.gmm_scene: GaussianMixture | None = None

    def subsample_pixels(self, pixels:np.ndarray) ->np.ndarray:
        if len(pixels)>self.max_samples:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(pixels), size=self.max_samples, replace=False)
            return pixels[indices]
        return pixels

    def fit_bg(self, converted_bg_img:np.ndarray) -> "DualGMM":
        "Fit the frozen reference Background GMM"
        pixels = converted_bg_img.reshape(-1, 3).astype(np.float64)
        sample_pixels = self.subsample_pixels(pixels)

        self.gmm_bg = GaussianMixture(
            n_components=self.k_bg,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=50,
            n_init = 3,
        )
        self.gmm_bg.fit(sample_pixels)
        return self

    def fit_bg_from_images(
            self,
            converted_bg_images: list[np.ndarray]
    )->"DualGMM":

        """
        Fit one universal bg GMM from multiple pre-processed empty-ws images.
        Each image contributes equal no of pixels.
        Args: 
            converted_bg_images: List of H*W*3 images in same color space and preprocessing format.
        """
        if not converted_bg_images:
            raise ValueError("At least one bg image is required.")
        n_images = len(converted_bg_images)
        if self.max_samples < n_images:
            raise ValueError(
                "max_samples must be at least the number of bg images."
            )
        rng = np.random.default_rng(self.random_state)
        samples_per_img = self.max_samples // n_images
        image_samples = []

        for i, image in enumerate(converted_bg_images):
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Background image {i} must have shape (H, W, 3)"
                )
            if image.shape[0] == 0 or image.shape[1] == 0:
                raise ValueError(
                    f"Bg image {i} is empty"
                )
            pixels = image.reshape(-1, 3).astype(np.float64)
            if not np.isfinite(pixels).all():
                raise ValueError(
                    f"Bg image {i} contains non-finite values"
                )
            n_samples = min(samples_per_img, len(pixels))
            indices = rng.choice(
                len(pixels),
                size = n_samples,
                replace=False
            )
            image_samples.append(pixels[indices])

        training_pixels = np.vstack(image_samples)
        if len(training_pixels) <self.k_bg:
            raise ValueError(
                f"Need at least {self.k_bg} training pixels",
                f"got {len(training_pixels)}"
            )

        self.gmm_bg = GaussianMixture(
            n_components=self.k_bg,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=50,
            n_init=3,
        )

        self.gmm_bg.fit(training_pixels)
        return self
    
    def fit_scene(self, converted_scene_img:np.ndarray) -> "DualGMM":
        pixels = converted_scene_img.reshape(-1, 3).astype(np.float64)
        sample_pixels = self.subsample_pixels(pixels)
        self.gmm_scene = GaussianMixture(
            n_components = self.k_scene,
            covariance_type = "full",
            random_state= self.random_state,
            max_iter=50,
            n_init=3,
        )
        self.gmm_scene.fit(sample_pixels)
        return self

    def compute_loglikelihood_comparison(self, converted_scene_img:np.ndarray) -> dict[str, np.ndarray]:
        """Computed pixel-wise log-likelihood under BG GMM, Scene GMM,
            and their log-likelihood ratio (contrast)
        """
        if self.gmm_bg is None or self.gmm_scene is None:
            raise RuntimeError("Both Background and Scene GMMs should be fitted")
        h, w = converted_scene_img.shape[:2]
        pixels = converted_scene_img.reshape(-1, 3).astype(np.float64)

        #1. Log likelihood of scene pixels under bg prior model
        bg_loglik = self.gmm_bg.score_samples(pixels).reshape(h,w)
        
        #2 Log-likelihood of scene pixels under scene own model
        scene_loglik = self.gmm_scene.score_samples(pixels).reshape(h,w)

        diff_loglik = scene_loglik - bg_loglik
        return {
            "bg_loglik": bg_loglik,
            "scene_loglik": scene_loglik,
            "diff_loglik": diff_loglik
        }