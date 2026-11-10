from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict
import os

@dataclass(frozen=True)
class PipelineConfig:
    target_size: Tuple[int, int]=(480,360)
    color_space: str = "rgb"
    k_bg: int = 5
    k_scene: int = 12
    max_gmm_samples: int= 25000
    bg_percentile: float=10
    fg_percentile: float=0.5
    grabcut_iterations: int=5
    border_margin: int=5
    project_root: Path = Path(__file__).resolve().parents[2]
    data_raw_dir: Path = project_root/"data"/"raw"
    data_out_dir: Path = project_root/"data"/"outputs"
    robot_ip: str = os.environ.get("NIRYO_IP", "192.168.24.7")
    niryo_poses: Dict[str, Tuple[float, ...]] = field(default_factory=lambda:{
        "P1": (-0.006, 0.213, -0.350, -0.048, -1.426, -0.061),
        "P2": (0.910, 0.025, 0.008, -0.532, -1.909, -0.121),
        "P3": (-0.845, -0.126, 0.111, 0.676, -1.905, -0.121),
        "P4": (0.026, -0.809, 0.456, -0.019, -1.911, -0.004),
        "P6": (-0.049, 0.026, -0.146, 0.006, -1.618, -0.062),
    })
    observation_pose: str = "P6"
    live_capture_poses : Tuple[str, ...] = ("P1","P2", "P3", "P4", "P6")
    captured_dir : Path = project_root/"data"/"captured_niryo"
    crop_method: str= "niryo"
    workspace_ratio: float =1.0
    workspace_side_px: int = 360

DEFAULT_CONFIG = PipelineConfig()
