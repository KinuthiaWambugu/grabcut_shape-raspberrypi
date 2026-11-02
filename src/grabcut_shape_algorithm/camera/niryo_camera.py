from pathlib import Path

import cv2
import numpy as np
from pyniryo import JointsPosition, NiryoRobot, uncompress_image


def _as_position(position):
    """Accept a JointsPosition or a plain 6-tuple of joint angles."""
    if isinstance(position, (tuple, list)):
        return JointsPosition(*position)
    return position


def connect(ip, intrinsics_path=None):
    """Connect, calibrate, and (optionally) save the camera intrinsics."""
    robot = NiryoRobot(ip)
    robot.calibrate_auto()

    # Get camera calibration parameters
    mtx, dist = robot.get_camera_intrinsics()
    if intrinsics_path is not None:
        intrinsics_path = Path(intrinsics_path)
        intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            intrinsics_path,
            camera_matrix=np.asarray(mtx),
            distortion_coefficients=np.asarray(dist),
        )
    return robot


def capture_pose(robot, label, position, output_dir, settle_s=1.5, filename=None):
    """Move to one pose, grab an image and save it. Returns the path, or None
    if the robot returned an invalid image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMoving to {label}: {position}")
    try:
        robot.move(_as_position(position))
        robot.wait(settle_s)

        img_compressed = robot.get_img_compressed()
        img = uncompress_image(img_compressed)

        if img is None or img.size == 0:
            print(f"Warning: Invalid image at {label}")
            return None

        image_path = output_dir / (filename or f"{label}.png")

        if cv2.imwrite(str(image_path), img):
            print(f"{label}: image saved to {image_path.resolve()}")
            return image_path
        print(f"{label}: failed to save image")
        return None

    except Exception as e:
        print(f"ERROR at {label}: {type(e).__name__}: {e}")
        print("Stopping the sequence because this pose failed.")
        raise


def capture_niryo_images(ip, positions, output_dir, capture_home=True,
                         settle_s=1.5, prefix=""):
    """Run the capture sequence. `positions` maps label -> joint position.

    Returns {label: Path} for every image that was saved.
    """
    output_dir = Path(output_dir)
    saved = {}
    robot = connect(ip, output_dir / "camera_intrinsics.npz")
    try:
        robot.move_to_home_pose()
        robot.wait(settle_s)

        for label, position in positions.items():
            path = capture_pose(
                robot, label, position, output_dir,
                settle_s=settle_s, filename=f"{prefix}{label}.png",
            )
            if path is not None:
                saved[label] = path

        if capture_home:
            # Return home and capture an additional image
            print("\nReturning to HOME...")
            robot.move_to_home_pose()
            robot.wait(settle_s)

            img = uncompress_image(robot.get_img_compressed())
            if img is not None and img.size > 0:
                home_path = output_dir / f"{prefix}HOME.png"
                if cv2.imwrite(str(home_path), img):
                    saved["HOME"] = home_path
                    print("HOME: image saved")

        print(f"\nFinished. Successfully saved {len(saved)} images.")
        return saved

    finally:
        robot.close_connection()
        print("Robot connection closed.")