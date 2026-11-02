import sys
import cv2
import argparse
import random

from src.grabcut_shape_algorithm.config import DEFAULT_CONFIG
from src.grabcut_shape_algorithm.core.gmm_priors import DualGMM
from src.grabcut_shape_algorithm.core.grabcut import GrabCutSegmenter
from src.grabcut_shape_algorithm.core.image_io import (
    load_detect_and_preprocess_image,
    load_and_preprocess_image,
)
from src.grabcut_shape_algorithm.utils.export import extract_cutout

def parse_args():
    parser = argparse.ArgumentParser(description="GrabCut step 1 on the Niryo workspace")
    parser.add_argument("--live-background", action="store_true", help="capture empty background images")
    parser.add_argument("--live-scene", action="store_true", help="capture scene with n objects")
    parser.add_argument("--ip", default=None, help="robot ip (defaults to one in config file, ROBOT_IP)" )
    parser.add_argument("--poses", nargs="+", default=None, metavar="LABEL", help="poses to capture from live poses")
    parser.add_argument("--seed", type=int, default=None, help="seed for random image obtaining")
    return parser.parse_args()

def acquire_paths(cfg, args):
    """
    Return bg and scene paths
    """
    bg_path = cfg.data_raw_dir/"Niryo"/"P6_crop.png"
    scene_path = cfg.data_raw_dir/"Niryo"/"scene01_P6_crop.png"
    if args.live_background or args.live_scene:
        from src.grabcut_shape_algorithm.camera.niryo_camera import (
            capture_pose,
            connect
        )
        labels = args.poses or list(cfg.live_capture_poses)
        unknown = [1 for 1 in labels if 1 not in cfg.niryo_poses]
        if unknown:
            print(f"[Error] Unknown pose labels: {unknown}. Known {list(cfg.niryo_poses)}")
            sys.exit(1)
        rng = random.Random(args.seed)

        def capture_set(robot, prefix):
            paths = []
            for label in labels:
                path = capture_pose(
                    robot, label, cfg.niryo_poses[label], cfg.captured_dir,
                    filename = f"{prefix}_{label}.png"
                )
                if path is not None:
                    paths.append(path)
            return paths

        robot = connect(
            args.ip or cfg.robot_ip,
            cfg.captured_dir/ "camera_intrinsics.npz"
        )
        try:
            robot.move_to_home_pose()
            robot.wait(1.5)
            if args.live_background:
                input("Clear workspace, press Enter to capture bg images:")
                bg_images = capture_set(robot, "background")
                if not bg_images:
                    print("Error no valid bg images was captured")
                    sys.exit(1)
                bg_path = rng.choice(bg_images)
                print(f"Background selected at random: {bg_path.name}"
                      f"from {len(bg_images)} captured")

            if args.live_scene:
                input("Place objects on scene and press Enter:")
                scene_images = capture_set(robot, "scene")
                if not scene_images:
                    print(f"Error, No valid scene images captured")
                    sys.exit(1)
                scene_path = rng.choice(scene_images)
                print(f"Scene selected at random: {scene_path.name}"
                      f"from {len(scene_images)} captured")
        finally:
            robot.close_connection()
            print("Robot disconnected")

    return bg_path, scene_path

def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.data_out_dir.mkdir(parents=True, exist_ok=True)

    bg_path, scene_path = acquire_paths(cfg, args)

    #bg_path = cfg.data_raw_dir / "Niryo/P6_crop.png"
    #scene_path = cfg.data_raw_dir / "Niryo/scene01_P6_crop.png"
    missing = [p for p in (bg_path, scene_path) if not p.exists()]
    if missing:
        print("Error, Missing inout images")
        for p in missing:
            print(f"{p}")
        sys.exit(1)

    """
        if not bg_path.exists() or not scene_path.exists():
            print(
                f"[ERROR] Place 'background_img.jpg' and "
                f"'Scene4.jpeg' in {cfg.data_raw_dir}"
            )
            sys.exit(1)
    """

    print("1. Loading & resizing images...")

    bg_bgr, bg_converted = load_and_preprocess_image(
        bg_path,
        target_size=cfg.target_size,
        color_space=cfg.color_space,
    )

    scene_cropped, bbox_raw, centres = (
        load_detect_and_preprocess_image(
            scene_path,
            target_size=cfg.target_size,
            color_space=cfg.color_space,
            detect_circles=True,
            debug=True,
        )
    )

    print(
        f"[✓] Background image: {bg_converted.shape}"
    )

    print(
        f"[✓] Scene cropped image: {scene_cropped.shape}"
    )

    print(
        f"[✓] Detected bounding box: {bbox_raw}"
    )

    print(
        f"[✓] Circle centres: {centres}"
    )

    # =========================================================
    # 2. Fit Background and Scene GMMs
    # =========================================================

    print(
        "\n2. Fitting Background GMM "
        f"(K={cfg.k_bg}) and Scene GMM (K=6)..."
    )

    analyzer = DualGMM(
        k_bg=cfg.k_bg,
        k_scene=12,
        max_samples=cfg.max_gmm_samples,
    )

    analyzer.fit_bg(bg_converted)
    analyzer.fit_scene(scene_cropped)

    print("[✓] GMM fitting complete.")

    # =========================================================
    # 3. Create raw GMM component-based seed masks
    # =========================================================

    print(
        "\n3. Creating GMM component-based seed masks..."
    )

    segmenter = GrabCutSegmenter(
        iterations=cfg.grabcut_iterations,
        border_margin=cfg.border_margin,
        morph_kernel_size=5,
    )

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

    print(
        f"[✓] Background Scene components: "
        f"{sorted(background_scene_components)}"
    )

    print(
        f"[✓] Foreground Scene components: "
        f"{sorted(foreground_scene_components)}"
    )

    # =========================================================
    # 4. Morphological cleanup and GrabCut initialization
    # =========================================================

    print(
        "\n4. Applying morphological cleanup "
        "and creating GrabCut trimap..."
    )

    seed_mask = segmenter.morphological_cleanup(
        raw_bg=raw_bg,
        raw_fg=raw_fg,
    )

    print("[✓] GrabCut seed mask created.")
    print("\n5. Executing GrabCut...")

    # run_grabcut() expects a BGR image.
    # scene_cropped is RGB when color_space="rgb".
    if cfg.color_space.lower() == "rgb":
        scene_bgr = cv2.cvtColor(
            scene_cropped,
            cv2.COLOR_RGB2BGR,
        )
    elif cfg.color_space.lower() == "bgr":
        scene_bgr = scene_cropped
    else:
        raise ValueError(
            "run_grabcut() requires a BGR/RGB image. "
            f"Received color_space='{cfg.color_space}'."
        )

    final_mask = segmenter.run_grabcut(
        scene_bgr,
        seed_mask,
    )

    print("[✓] GrabCut segmentation complete.")

    # =========================================================
    # 6. Export outputs
    # =========================================================

    print("\n6. Exporting outputs...")

    binary_mask, rgba_cutout = extract_cutout(
        scene_bgr,
        final_mask,
    )

    cv2.imwrite(
        str(cfg.data_out_dir / "step1_mask.png"),
        binary_mask,
    )

    cv2.imwrite(
        str(cfg.data_out_dir / "step1_cutout.png"),
        rgba_cutout,
    )

    print(
        "✓ Done. Check results in "
        "data/outputs/"
    )


if __name__ == "__main__":
    main()