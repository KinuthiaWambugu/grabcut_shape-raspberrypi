import os
import cv2
from pyniryo import *

# Robot configuration
ROBOT_IP = "192.168.24.7"

SAVE_DIR = "data/gate1_captures"
SETTLE_TIME = 1.5

# Your existing joint positions
POSITIONS = {
    "P1": (-0.009, -0.064, -0.167, -0.019, -1.560, -0.032),
    "P2": (0.659, -0.211, -0.217, -0.489, -1.652, -0.506),
    "P3": (-0.545, -0.553, 0.088, 0.681, -1.816, 0.009),
}


def save_image(save_dir, name, img):
    """Save a captured image."""
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, name)

    if cv2.imwrite(path, img):
        print(f"  -> Saved: {path}")
    else:
        print(f"  -> Failed to save: {path}")


def capture_image(robot, label):
    """Capture and save an image at the current robot position."""

    img_compressed = robot.get_img_compressed()
    img = uncompress_image(img_compressed)

    if img is None or img.size == 0:
        raise RuntimeError(f"Invalid camera image at {label}")

    save_image(SAVE_DIR, f"{label}.png", img)

    return img


def move_and_capture(robot, label, joints):
    """Move to a joint position, wait, and capture an image."""

    print(f"\nMoving to {label}: {joints}")

    robot.move(JointsPosition(*joints))
    robot.wait(SETTLE_TIME)

    capture_image(robot, label)


def run_gate1():
    robot = None

    try:
        # 1. Connect
        print("Connecting to Niryo Ned 2...")
        robot = NiryoRobot(ROBOT_IP)

        # 2. Calibrate
        print("Calibrating robot...")
        robot.calibrate_auto()

        # 3. Move to HOME
        print("Moving to HOME...")
        robot.move_to_home_pose()
        robot.wait(SETTLE_TIME)

        # 4. Visit each observation position
        for label, joints in POSITIONS.items():
            move_and_capture(robot, label, joints)

        # 5. Return HOME
        print("\nReturning to HOME...")
        robot.move_to_home_pose()
        robot.wait(SETTLE_TIME)

        # 6. Capture HOME image
        capture_image(robot, "HOME")

        print("\nGate 1 capture sequence completed.")

    except Exception as e:
        print(f"\nGate 1 failed: {type(e).__name__}: {e}")
        raise

    finally:
        if robot is not None:
            robot.close_connection()
            print("Robot connection closed.")


if __name__ == "__main__":
    run_gate1()
