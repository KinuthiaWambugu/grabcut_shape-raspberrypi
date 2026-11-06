"""Capture background and scene batches from the Niryo Ned 2 camera.

    python capture_dataset.py

1. Background: with the workspace EMPTY, every pose (and HOME) is photographed.
   Images go to cfg.background_dir. You are then asked whether to capture another
   background batch (change the lighting a little for a more robust model).
2. Scenes: set up objects, capture, repeat. Images go to cfg.scene_dir.

At every prompt press Enter to capture, or q to quit. Batches are numbered
(background01_P1.png, scene02_P1.png, ...) and never overwrite earlier ones.
With cropping on (default) each folder also gets crops/ and markers/ subfolders.
"""
import argparse
import sys

from src.grabcut_shape_algorithm.camera.niryo_camera import capture_session, connect
from src.grabcut_shape_algorithm.config import DEFAULT_CONFIG as cfg


def parse_args():
    p = argparse.ArgumentParser(description="Capture background and scene batches")
    p.add_argument("--ip", default=None, help="robot IP (defaults to NIRYO_IP / cfg.robot_ip)")
    p.add_argument("--poses", nargs="+", default=None, metavar="LABEL",
                   help="poses to visit (default: cfg.live_capture_poses)")
    p.add_argument("--no-crop", action="store_true", help="do not save marker crops/overlays")
    p.add_argument("--no-home", action="store_true", help="skip the HOME photo at the end of each batch")
    return p.parse_args()


def main():
    args = parse_args()
    labels = args.poses or list(cfg.live_capture_poses)
    unknown = [lbl for lbl in labels if lbl not in cfg.niryo_poses]
    if unknown:
        print(f"[ERROR] Unknown pose label(s): {unknown}. Known: {list(cfg.niryo_poses)}")
        sys.exit(1)
    positions = {lbl: cfg.niryo_poses[lbl] for lbl in labels}

    robot = connect(args.ip or cfg.robot_ip, cfg.captured_dir / "camera_intrinsics.npz")
    try:
        result = capture_session(
            robot, positions, cfg.background_dir, cfg.scene_dir,
            capture_home=not args.no_home,
            crop=not args.no_crop,
            workspace_ratio=cfg.workspace_ratio,
            workspace_side_px=cfg.workspace_side_px,
        )
    finally:
        robot.close_connection()
        print("Robot disconnected")

    n_bg, n_sc = len(result["background"]), len(result["scene"])
    print(f"\nDone: {n_bg} background batch(es) in {cfg.background_dir}, "
          f"{n_sc} scene batch(es) in {cfg.scene_dir}")
    if n_bg:
        print("Next: fit the background model from the background folder.")


if __name__ == "__main__":
    main()