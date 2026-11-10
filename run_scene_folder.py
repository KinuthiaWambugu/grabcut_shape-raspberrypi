import sys
import traceback
from pathlib import Path

import cv2

from src.grabcut_shape_algorithm.config import DEFAULT_CONFIG
from src.grabcut_shape_algorithm.core.gmm_priors import DualGMM
from src.grabcut_shape_algorithm.core.grabcut import GrabCutSegmenter
from src.grabcut_shape_algorithm.core.image_io import (
    load_detect_and_preprocess_image,
    load_and_preprocess_image,
)
from src.grabcut_shape_algorithm.utils.export import extract_cutout

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_scene_images(scenes_dir: Path):
    return sorted(
        p for p in scenes_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_scene(scene_path, cfg, analyzer, segmenter):
    """Run the full pipeline on one scene image and save its outputs."""
    stem = scene_path.stem

    # 1. Load scene
    scene_cropped, bbox_raw, centres = load_detect_and_preprocess_image(
        scene_path,
        target_size=cfg.target_size,
        color_space=cfg.color_space,
        detect_circles=True,
        debug=False,  # avoid popping up debug windows for every image
    )
    print(f"    [✓] Cropped scene: {scene_cropped.shape}, bbox: {bbox_raw}")
    print(f"    [✓] Circle centres: {centres}")

    # 2. Fit scene GMM (background GMM is already fitted)
    analyzer.fit_scene(scene_cropped)

    # 3. Seed masks from GMM components
    (
        raw_bg,
        raw_fg,
        component_labels,
        component_confidence,
        background_scene_components,
        foreground_scene_components,
        valid_matches,
        cost_matrix,
    ) = segmenter.create_seed_mask(
        scene_img=scene_cropped,
        gmm=analyzer,
        max_color_distance=100.0,
    )
    print(f"    [✓] BG components: {sorted(background_scene_components)}")
    print(f"    [✓] FG components: {sorted(foreground_scene_components)}")

    # 4. Morphological cleanup
    seed_mask = segmenter.morphological_cleanup(raw_bg=raw_bg, raw_fg=raw_fg)

    # 5. GrabCut (expects BGR)
    color_space = cfg.color_space.lower()
    if color_space == "rgb":
        scene_bgr = cv2.cvtColor(scene_cropped, cv2.COLOR_RGB2BGR)
    elif color_space == "bgr":
        scene_bgr = scene_cropped
    else:
        raise ValueError(
            "run_grabcut() requires a BGR/RGB image. "
            f"Received color_space='{cfg.color_space}'."
        )

    final_mask = segmenter.run_grabcut(scene_bgr, seed_mask)

    # 6. Export
    binary_mask, rgba_cutout = extract_cutout(scene_bgr, final_mask)
    cv2.imwrite(str(cfg.data_out_dir / f"{stem}_mask.png"), binary_mask)
    cv2.imwrite(str(cfg.data_out_dir / f"{stem}_cutout.png"), rgba_cutout)
    print(f"    [✓] Saved {stem}_mask.png and {stem}_cutout.png")


def main():
    cfg = DEFAULT_CONFIG
    cfg.data_out_dir.mkdir(parents=True, exist_ok=True)

    bg_path = cfg.data_raw_dir / "Niryo bg/P2.png"

    # Scenes folder: first CLI argument, else <data_raw_dir>/scenes
    scenes_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.data_raw_dir / "Scene"

    if not bg_path.exists():
        print(f"[ERROR] Place 'background_img.jpg' in {cfg.data_raw_dir}")
        sys.exit(1)

    if not scenes_dir.is_dir():
        print(f"[ERROR] Scenes folder not found: {scenes_dir}")
        sys.exit(1)

    scene_paths = find_scene_images(scenes_dir)
    if not scene_paths:
        print(f"[ERROR] No images found in {scenes_dir}")
        sys.exit(1)

    # =========================================================
    # One-time setup: background image + background GMM
    # =========================================================
    print("1. Loading background image...")
    bg_bgr, bg_converted = load_and_preprocess_image(
        bg_path,
        target_size=cfg.target_size,
        color_space=cfg.color_space,
    )
    print(f"[✓] Background image: {bg_converted.shape}")

    print(f"\n2. Fitting Background GMM (K={cfg.k_bg})...")
    analyzer = DualGMM(
        k_bg=cfg.k_bg,
        k_scene=cfg.k_scene,
        max_samples=cfg.max_gmm_samples,
    )
    analyzer.fit_bg(bg_converted)
    print("[✓] Background GMM fitted.")

    segmenter = GrabCutSegmenter(
        iterations=cfg.grabcut_iterations,
        border_margin=cfg.border_margin,
        morph_kernel_size=5,
    )

    # =========================================================
    # Per-scene processing
    # =========================================================
    print(f"\n3. Processing {len(scene_paths)} scene(s) from {scenes_dir}")

    succeeded, failed = [], []
    for i, scene_path in enumerate(scene_paths, start=1):
        print(f"\n[{i}/{len(scene_paths)}] {scene_path.name}")
        try:
            process_scene(scene_path, cfg, analyzer, segmenter)
            succeeded.append(scene_path.name)
        except Exception as e:
            print(f"    [✗] Failed: {e}")
            traceback.print_exc()
            failed.append(scene_path.name)

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n✓ Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("  Failed images: " + ", ".join(failed))
    print(f"  Results in {cfg.data_out_dir}")


if __name__ == "__main__":
    main()