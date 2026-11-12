from pyniryo import NiryoRobot, ObjectShape, ObjectColor

from src.grabcut_shape_algorithm.config import DEFAULT_CONFIG as cfg
from src.grabcut_shape_algorithm.robot.niryo_vision import detect_object


# IMPORTANT: Replace with the exact workspace name saved in Niryo Studio.
WORKSPACE_NAME = "workspace_1"

HEIGHT_OFFSET = 0.0
SETTLE_TIME = 1.5


def main():
    robot = NiryoRobot(cfg.robot_ip)

    try:
        print("Connecting to Niryo Ned 2...")
        robot.calibrate_auto()

        # Activate/update the currently attached tool.
        robot.update_tool()

        observation_joints = cfg.niryo_poses[cfg.observation_pose]

        result = detect_object(
            robot=robot,
            workspace_name=WORKSPACE_NAME,
            observation_joints=observation_joints,
            height_offset=HEIGHT_OFFSET,
            shape=ObjectShape.ANY,
            color=ObjectColor.ANY,
            settle_time=SETTLE_TIME,
        )

        object_found, object_pose, detected_shape, detected_color = result

        if object_found:
            print("\nGate 1 detection completed successfully.")
            print("Detected pose is available for the next gate.")
        else:
            print("\nGate 1 completed, but no object was detected.")

    finally:
        robot.close_connection()
        print("Robot connection closed.")


if __name__ == "__main__":
    main()
