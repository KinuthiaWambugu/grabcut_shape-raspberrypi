import argparse
from pathlib import Path

import cv2
import numpy as np

from src.grabcut_shape_algorithm.core.universal_bg import UniversalBackgroundGMM


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an already-cropped workspace image and convert BGR to RGB."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def parse_pose_image(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected POSE=PATH, e.g. P1=data/raw/Niryo/background/P1.png"
        )
    pose_id, image_path = value.split("=", 1)
    pose_id, image_path = pose_id.strip(), image_path.strip()
    if not pose_id or not image_path:
        raise argparse.ArgumentTypeError("Both pose ID and image path are required")
    return pose_id, Path(image_path)


def fit_universal_bg(
    pose_image_paths: dict[str, list[Path]],
    output_path: Path,
    n_components: int = 5,
    max_samples_per_pose: int = 25000,
    random_state: int = 42,
) -> UniversalBackgroundGMM:
    images_by_pose = {
        pose_id: [load_rgb_image(path) for path in paths]
        for pose_id, paths in pose_image_paths.items()
    }

    model = UniversalBackgroundGMM(
        n_components=n_components,
        max_samples_per_pose=max_samples_per_pose,
        random_state=random_state,
    ).fit(images_by_pose)
    model.save(output_path)

    print("\\nUNIVERSAL BACKGROUND GMM")
    print("-" * 60)
    print(f"Poses: {', '.join(model.pose_ids)}")
    print(f"Components: {model.n_components}")
    print(f"Samples per pose: {model.training_samples_per_pose}")
    print(f"Saved model: {output_path}")

    print("\\nPer-pose training-image likelihoods")
    for pose_id, images in images_by_pose.items():
        scores = np.concatenate(
            [model.score_samples(image).ravel() for image in images]
        )
        print(
            f"{pose_id}: mean={scores.mean():.3f}, "
            f"p05={np.percentile(scores, 5):.3f}, "
            f"median={np.median(scores):.3f}, "
            f"p95={np.percentile(scores, 95):.3f}"
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit a shared background GMM from multiple Niryo poses"
    )
    parser.add_argument(
        "--image", action="append", type=parse_pose_image, required=True,
        help="Cropped empty-workspace image as POSE=PATH; repeat per image",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/models/universal_background_gmm.joblib"),
    )
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--samples-per-pose", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths_by_pose: dict[str, list[Path]] = {}
    for pose_id, image_path in args.image:
        paths_by_pose.setdefault(pose_id, []).append(image_path)

    fit_universal_bg(
        paths_by_pose,
        args.output,
        n_components=args.components,
        max_samples_per_pose=args.samples_per_pose,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
