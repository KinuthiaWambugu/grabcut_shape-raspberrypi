from pyniryo import ObjectColor, ObjectShape, JointsPosition

def detect_object(
    robot,
    workspace_name: str,
    observation_joints: tuple,
    height_offset: float = 0.0,
    shape=ObjectShape.ANY,
    color=ObjectColor.ANY,
    settle_time: float = 1.5,
):
    """
    Move to the observation position and detect an object
    using Niryo's native vision API.

    Returns:
        object_found, object_pose, detected_shape, detected_color
    """

    if len(observation_joints) != 6:
        raise ValueError("observation_joints must contain six joint values")

    print("\nMoving to observation position...")
    robot.move(JointsPosition(*observation_joints))
    robot.wait(settle_time)

    print(f"Detecting objects in workspace: {workspace_name}")

    result = robot.get_target_pose_from_cam(
        workspace_name,
        height_offset=height_offset,
        shape=shape,
        color=color,
    )

    object_found, object_pose, detected_shape, detected_color = result

    print("\n--- Niryo Vision Detection ---")
    print(f"Object found: {object_found}")

    if object_found:
        print(f"Object pose: {object_pose}")
        print(f"Object shape: {detected_shape}")
        print(f"Object color: {detected_color}")
    else:
        print("No matching object detected.")

    return result
