from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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

DEFAULT_CONFIG = PipelineConfig()
