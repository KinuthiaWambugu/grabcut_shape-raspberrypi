import sys
import cv2
import argparse
import random
from pathlib import Path

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
    parser.add_argument("--bg", default=None, help="background image path (overrides the default offline file)")
    parser.add_argument("--scene", default=None, help="scene image path (overrides the default offline file)")
    parser.add_argument("--crop", action="store_true",
                        help="offline inputs are full Niryo frames: crop them to the 4-marker workspace first")
    return parser.parse_args()

def use_niryo_crop(cfg)->bool:
    return getattr(cfg, "crop_method", "niryo") == "niryo"

def crop_to_workspace(image_path, cfg):
    """Full Niryo frame -> workspace crop between the 4 markers.

    Saves the crop to <dir>/crops/<name> and the marker overlay to <dir>/markers/<name>
    (same layout as crop_path_for / markers_path_for). Returns the crop path.
    Raises WorkspaceNotFound if the markers can't be found.
    """
    from src.grabcut_shape_algorithm.camera.niryo_camera import (
        crop_workspace,
        crop_path_for,
        markers_path_for,
    )

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"OpenCV failed to read {image_path}")

    ws = crop_workspace(img, cfg.workspace_ratio, cfg.workspace_side_px)

    crop_path = crop_path_for(image_path)
    markers_path = markers_path_for(image_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    markers_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(markers_path), ws.debug_image)
    if not cv2.imwrite(str(crop_path), ws.image):
        raise IOError(f"Failed to save workspace crop to {crop_path}")

    flag = "" if ws.debug_ok else "  (low-contrast markers: debug_markers disagrees)"
    print(f"{image_path.name}: workspace crop {ws.image.shape[1]}x{ws.image.shape[0]} -> {crop_path}{flag}")
    return crop_path


def acquire_paths(cfg, args):
    """
    Return bg and scene paths
    """
    bg_path = Path(args.bg) if args.bg else cfg.data_raw_dir/"Niryo"/"P6_crop.png"
    scene_path = Path(args.scene) if args.scene else cfg.data_raw_dir/"Niryo"/"scene01_P6_crop.png"
    do_crop = use_niryo_crop(cfg)

    if args.crop and do_crop and not(args.live_background or args.live_scene):
        from src.grabcut_shape_algorithm.camera.niryo_camera import WorkspaceNotFound
        try:
            bg_path = crop_to_workspace(bg_path, cfg)
            scene_path = crop_to_workspace(scene_path, cfg)
        except WorkspaceNotFound as e:
            print(f"[Error]: {e}")
            sys.exit(1)

    if args.live_background or args.live_scene:
        from src.grabcut_shape_algorithm.camera.niryo_camera import (
            capture_pose,
            connect,
            WorkspaceNotFound
        )
        labels = args.poses or list(cfg.live_capture_poses)
        unknown = [l for l in labels if l not in cfg.niryo_poses]
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
                if path is None:
                    continue
                if do_crop:
                    try:
                        path = crop_to_workspace(path, cfg)
                    except WorkspaceNotFound as e:
                        print(f"{prefix}_{label}: {e}-skipping pose")
                        continue
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

    print("1. Loading & resizing images...")

    bg_bgr, bg_converted = load_and_preprocess_image(
        bg_path,
        target_size=cfg.target_size,
        color_space=cfg.color_space,
    )
    """
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
    """
    if use_niryo_crop(cfg):
        _, scene_cropped = load_and_preprocess_image(
            scene_path,
            target_size=cfg.target_size,
            color_space=cfg.color_space,
        )
        bbox_raw, centres = None, None
    else:
        scene_cropped, bbox_raw, centres = load_detect_and_preprocess_image(
            scene_path,
            target_size=cfg.target_size,
            color_space=cfg.color_space,
            detect_circles=True,
            debug=True,
        )

    print(f"[✓] Background image: {bg_converted.shape}")
    print(f"[✓] Scene cropped image: {scene_cropped.shape}")
    if bbox_raw is not None:
        print(f"[✓] Detected bounding box: {bbox_raw}")
        print(f"[✓] Circle centres: {centres}")


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