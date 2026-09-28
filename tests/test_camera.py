from pathlib import Path
from src.grabcut_shape_algorithm.camera.rasp_camera import(RaspberryCamera)

def main():
    output_dir = Path("data/captured")
    camera = RaspberryCamera(
        width=1280,
        height=720
    )
    try:
        print("Starting camera")
        camera.start_camera()
        output_path = camera.capture(
            output_dir/ "test_001.jpg"
        )
        print(f"Image captured: {output_path}")
    except Exception as e:
        print(f"Camera error: {e}")

    finally:
        camera.stop()
        print("Camera stopped")

if __name__ == "__main__":
    main()