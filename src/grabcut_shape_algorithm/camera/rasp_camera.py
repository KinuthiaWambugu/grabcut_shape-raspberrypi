from pathlib import Path
import time
from picamera2 import PiCamera2 #only on linux

class RaspberryCamera:
    def __init__(
            self,
            width: int=1280,
            height: int=720,):
        self.width = width
        self.height = height
        self.camera = PiCamera2()
        config = self.camaera.create_still_configuration(
            main={
                "size":(self.width, self.height),
                "format":"RGB888"
            }
        )
        self.camera.configure(config)

    def start_camera(self)-> None:
        self.camera.start()
        time.sleep(2)

    def capture(self, output_path: Path |str)->Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self.camera.capture_file(str(output_path))
        return output_path

    def stop(self)->None:
        self.camera.stop()
