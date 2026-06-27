from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DEMO_DIR = DATA_DIR / "demo"
LITERATURE_DIR = DATA_DIR / "literature"
LITERATURE_UPLOADS_DIR = LITERATURE_DIR / "uploads"

for path in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, DEMO_DIR, LITERATURE_DIR, LITERATURE_UPLOADS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    tolerance: int = 2
    sigma: float = 1.2
    canny_low: int = 60
    canny_high: int = 160
    log_sigma: float = 1.2
    morph_kernel: int = 3


@dataclass
class LLMConfig:
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
