from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    default_dtype: str = "float16"  # "float16" | "bfloat16" | "float32"


CONFIG = ProjectConfig()


