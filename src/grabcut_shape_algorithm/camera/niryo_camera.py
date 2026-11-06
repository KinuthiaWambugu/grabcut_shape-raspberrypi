from pathlib import Path
from dataclasses import dataclass
import re

import cv2
import numpy as np
from pyniryo import (
    JointsPosition, 
    NiryoRobot, 
    uncompress_image, 
    debug_markers,
    extract_img_workspace
    )


def _as_position(position):
    """Accept a JointsPosition or a plain 6-tuple of joint angles."""
    if isinstance(position, (tuple, list)):
        return JointsPosition(*position)
    return position

class WorkspaceNotFound(RuntimeError):
    "The 4 Niryo markers couldn't be detected in the image"

@dataclass
class WorkspaceCrop:
    image: np.ndarray #Corrected area between the markers
    debug_image: np.ndarray #full frame with detected markers drawn and numbered
    debug_ok: bool

def set_workspace_res(side_px):
    try:
        from pyniryo.vision import markers_detection
        markers_detection.IM_EXTRACT_SMALL_SIDE_PIXELS = int(side_px)
    except Exception as e:
        print(f"Warning: couldn't set working res {e}; using pyNiryo default")

def crop_workspace(img_bgr, workspace_ratio=1.0, side_px=360):
    set_workspace_res(side_px)
    debug_ok, debug_image = debug_markers(img_bgr, workspace_ratio)
    crop = extract_img_workspace(img_bgr, workspace_ratio)
    if crop is None or crop.size == 0:
        raise WorkspaceNotFound("Could not find 4 Niryo markers")
    return WorkspaceCrop(image=crop, debug_image=debug_image, debug_ok=bool(debug_ok))

def crop_path_for(image_path):
    image_path = Path(image_path)
    return image_path.parent / "crops" / image_path.name
 
 
def markers_path_for(image_path):
    image_path = Path(image_path)
    return image_path.parent / "markers" / image_path.name

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

def capture_here(robot, label, output_dir, filename=None, crop=False,
                 workspace_ratio=1.0, workspace_side_px=360):

    """
    Grab one image at the arm's CURRENT pose and save it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_compressed = robot.get_img_compressed()
    img = uncompress_image(img_compressed)
    if img is None or img.size == 0:
        print(f"Warning: Invalid image at {label}")
        return None

    image_path = output_dir / (filename or f"{label}.png")
    if not cv2.imwrite(str(image_path), img):
        print(f"{label}: failed to save image")
        return None
    print(f"{label}: image saved to {image_path.resolve()}")

    if not crop:
        return image_path

    try:
        ws = crop_workspace(img, workspace_ratio, workspace_side_px)
    except WorkspaceNotFound as e:
        print(f"{label}: {e}")
        return None

    crop_path = crop_path_for(image_path)
    markers_path = markers_path_for(image_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    markers_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(markers_path), ws.debug_image)
    if not cv2.imwrite(str(crop_path), ws.image):
        print(f"{label}: failed to save workspace crop")
        return None
    flag = "" if ws.debug_ok else " (low-contrast markers: debug_markers disagrees)"
    print(f"{label}: workspace crop {ws.image.shape[1]}x{ws.image.shape[0]} "
            f"saved to {crop_path.resolve()}{flag}")
    return crop_path


def capture_pose(robot, label, position, output_dir, settle_s=1.5, filename=None,
                 crop=False, workspace_ratio=1.0, workspace_side_px=360):
    """Move to one pose, grab an image and save it. Returns the path, or None
    if the robot returned an invalid image."""
    print(f"\n Moving to {label}:{position}")
    try:
        robot.move(_as_position(position))
        robot.wait(settle_s)
        return capture_here(
            robot, label, output_dir, filename=filename, crop=crop,
            workspace_ratio=workspace_ratio, workspace_side_px=workspace_side_px
        )
    except Exception as e:
        print(f"ERROR at {label}: {type(e).__name__}: {e}")
        print("Stopping the sequence because this pose failed.")
        raise

def capture_batch(robot, positions, save_dir, prefix="", capture_home=True,
                  settle_s=1.5, crop=False, workspace_ratio=1.0, workspace_side_px=360):
    """Visit every pose in `positions` (label -> joints), then HOME, saving one image
    per stop into `save_dir`. Uses an already connected robot.
 
    Returns {label: Path} for every image that was saved. If a move fails the
    exception propagates and the sequence stops (the arm state is then unknown).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
 
    robot.move_to_home_pose()
    robot.wait(settle_s)
 
    for label, position in positions.items():
        path = capture_pose(
            robot, label, position, save_dir,
            settle_s=settle_s, filename=f"{prefix}{label}.png",
            crop=crop, workspace_ratio=workspace_ratio,
            workspace_side_px=workspace_side_px,
        )
        if path is not None:
            saved[label] = path
 
    if capture_home:
        print("\nReturning to HOME...")
        robot.move_to_home_pose()
        robot.wait(settle_s)
        path = capture_here(
            robot, "HOME", save_dir, filename=f"{prefix}HOME.png", crop=crop,
            workspace_ratio=workspace_ratio, workspace_side_px=workspace_side_px,
        )
        if path is not None:
            saved["HOME"] = path
 
    print(f"\nCaptured {len(saved)} images in '{save_dir}': {list(saved)}")
    return saved

def ask_continue(prompt, input_fn=None):
    """Ask whether to capture. Returns True to go on, False to quit.
 
    Enter / y / yes -> True.  q / quit / n / no (also Ctrl-C or end of input) -> False.
    Anything else asks again, so a stray key never moves the arm by accident.
    """
    read = input_fn or input   # looked up at call time, so tests can replace input()
    while True:
        try:
            answer = read(f"{prompt} [Enter = capture, q = quit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer in ("", "y", "yes"):
            return True
        if answer in ("q", "quit", "n", "no"):
            return False
        print("  Press Enter to capture, or type q to quit.")
 
 
def next_batch_index(save_dir, stem):
    """First free batch number in `save_dir` for files named <stem>NN_*, so repeated
    runs add new batches instead of overwriting old ones."""
    pattern = re.compile(rf"^{re.escape(stem)}(\d+)_")
    used = [int(m.group(1)) for f in Path(save_dir).glob("*.png")
            if (m := pattern.match(f.name))] if Path(save_dir).is_dir() else []
    return max(used, default=0) + 1
 
 
def capture_batches(robot, positions, save_dir, stem, first_prompt, next_prompt,
                    ask=ask_continue, **batch_kwargs):
    """Capture numbered batches (<stem>01_P1.png, <stem>02_P1.png, ...) into
    `save_dir` for as long as the user keeps answering yes.
 
    Returns a list with one {label: Path} dict per batch (empty if the user quit
    straight away).
    """
    batches = []
    prompt = first_prompt
    while ask(prompt):
        idx = next_batch_index(save_dir, stem)
        print(f"\n=== Capturing {stem.upper()} batch {idx:02d} ===")
        batches.append(
            capture_batch(robot, positions, save_dir, prefix=f"{stem}{idx:02d}_", **batch_kwargs)
        )
        prompt = next_prompt
    return batches
 
 
def capture_session(robot, positions, background_dir, scene_dir, ask=ask_continue, **batch_kwargs):
    """Interactive session: background batches first, then scene batches.
 
    Background images go to `background_dir`, scene images to `scene_dir`. After
    each batch the user is asked whether to capture another one.
    Returns {"background": [batch, ...], "scene": [batch, ...]}.
    """
    print("=== BACKGROUND (empty workspace) ===")
    background = capture_batches(
        robot, positions, background_dir, "background",
        "Clear the workspace, then capture the background",
        "Capture another background batch (e.g. different lighting)?",
        ask=ask, **batch_kwargs,
    )
    print("\n=== SCENES (objects on the workspace) ===")
    scenes = capture_batches(
        robot, positions, scene_dir, "scene",
        "Set up the scene, then capture",
        "Set up another scene and capture?",
        ask=ask, **batch_kwargs,
    )
    return {"background": background, "scene": scenes}
 
def  capture_niryo_images(ip, positions, output_dir, capture_home=True,
                         settle_s=1.5, prefix="", crop=False, workspace_ratio=1.0, workspace_side_px=360):
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
                crop=crop, workspace_ratio=workspace_ratio,
                workspace_side_px=workspace_side_px
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


