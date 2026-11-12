from unittest.mock import Mock

from src.grabcut_shape_algorithm.robot.niryo_vision import detect_object


def test_detect_object_calls_niryo_vision():
    robot = Mock()

    expected_result = (
        True,
        "mock_pose",
        "CIRCLE",
        "RED",
    )

    robot.get_target_pose_from_cam.return_value = expected_result

    joints = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    result = detect_object(
        robot=robot,
        workspace_name="workspace_1",
        observation_joints=joints,
    )

    assert result == expected_result

    robot.move.assert_called_once()
    robot.wait.assert_called_once_with(1.5)

    robot.get_target_pose_from_cam.assert_called_once_with(
        "workspace_1",
        height_offset=0.0,
        shape=robot.get_target_pose_from_cam.call_args.kwargs["shape"],
        color=robot.get_target_pose_from_cam.call_args.kwargs["color"],
    )
