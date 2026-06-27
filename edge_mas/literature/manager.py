from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

from edge_mas.config import LITERATURE_DIR, LITERATURE_UPLOADS_DIR
from edge_mas.utils import safe_filename


@dataclass
class LiteratureRecord:
    topic: str
    title: str
    authors: str
    year: str
    source: str
    note: str
    keywords: str


SEED_LITERATURE = [
    LiteratureRecord(
        topic="Canny",
        title="A Computational Approach to Edge Detection",
        authors="John Canny",
        year="1986",
        source="IEEE Transactions on Pattern Analysis and Machine Intelligence",
        note="经典 Canny 边缘检测框架，包括平滑、梯度、非极大值抑制和双阈值连接思想。",
        keywords="Canny; edge detection; non-maximum suppression; hysteresis",
    ),
    LiteratureRecord(
        topic="Sobel / Prewitt",
        title="Gradient-based edge operators",
        authors="Sobel; Prewitt and related early operators",
        year="1960s-1970s",
        source="Classical image processing literature",
        note="一阶梯度边缘检测思想，计算简单，常作为传统边缘检测基线。",
        keywords="Sobel; Prewitt; gradient; first-order derivative",
    ),
    LiteratureRecord(
        topic="LoG",
        title="Theory of Edge Detection",
        authors="David Marr; Ellen Hildreth",
        year="1980",
        source="Proceedings of the Royal Society B",
        note="基于高斯平滑和二阶导数零交叉的边缘检测思想，是 LoG 方法的重要理论来源。",
        keywords="LoG; zero crossing; Laplacian of Gaussian; Marr-Hildreth",
    ),
    LiteratureRecord(
        topic="Morphological Gradient",
        title="Morphological image analysis foundations",
        authors="Jean Serra; related mathematical morphology researchers",
        year="1980s",
        source="Mathematical morphology literature",
        note="形态学梯度利用膨胀与腐蚀差分提取结构边界，适合展示非微分型边缘检测思想。",
        keywords="mathematical morphology; dilation; erosion; morphological gradient",
    ),
    LiteratureRecord(
        topic="Deep Learning Edge Detection",
        title="Holistically-Nested Edge Detection",
        authors="Saining Xie; Zhuowen Tu",
        year="2015",
        source="IEEE International Conference on Computer Vision",
        note="深度学习边缘检测代表性工作之一，用于说明边缘检测从手工算子到学习型方法的发展。",
        keywords="HED; deep learning; edge detection; CNN",
    ),
    LiteratureRecord(
        topic="Evaluation",
        title="Boundary detection evaluation with tolerance",
        authors="Berkeley segmentation benchmark and related evaluation studies",
        year="2000s",
        source="Image segmentation and boundary detection evaluation literature",
        note="边缘位置可能存在轻微偏移，因此常采用带容差的 Precision、Recall 和 F-measure。",
        keywords="Precision; Recall; F-measure; boundary matching; tolerance",
    ),
]


def literature_csv_path() -> Path:
    return LITERATURE_DIR / "literature_records.csv"


def literature_jsonl_path() -> Path:
    return LITERATURE_DIR / "literature_records.jsonl"


def save_seed_literature() -> tuple[Path, Path]:
    LITERATURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = literature_csv_path()
    jsonl_path = literature_jsonl_path()

    fieldnames = list(asdict(SEED_LITERATURE[0]).keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in SEED_LITERATURE:
            writer.writerow(asdict(rec))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in SEED_LITERATURE:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def load_literature_records() -> list[dict]:
    csv_path = literature_csv_path()
    if not csv_path.exists():
        save_seed_literature()
    records: list[dict] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def save_uploaded_literature(files) -> list[Path]:
    LITERATURE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files or []:
        path = LITERATURE_UPLOADS_DIR / safe_filename(file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(path)
    return saved


def append_manual_record(record: LiteratureRecord) -> tuple[Path, Path]:
    records = load_literature_records()
    records.append(asdict(record))
    fieldnames = list(asdict(record).keys())
    csv_path = literature_csv_path()
    jsonl_path = literature_jsonl_path()
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def create_literature_summary(target_path: Path, task: str, selected_algorithms: list[str]) -> Path:
    records = load_literature_records()
    algos = "、".join(selected_algorithms)
    lines = [
        "# Step 1 文献调研智能体输出：文献调研摘要",
        "",
        f"## 研究任务",
        task,
        "",
        "## 本次实验涉及的公开边缘检测方法",
        algos or "未指定",
        "",
        "## 文献调研结论",
        "文献调研智能体围绕传统边缘检测算子、多阶段 Canny 框架、形态学梯度、深度学习边缘检测以及边界评价指标整理了种子文献。该产物用于支撑后续算法设计智能体生成算法对比方案。",
        "",
        "## 已保存文献记录",
    ]
    for idx, row in enumerate(records, start=1):
        lines.append(f"{idx}. **{row.get('title','')}** ({row.get('year','')}) — {row.get('note','')}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(lines), encoding="utf-8")
    return target_path


def copy_literature_records_to_run(run_dir: Path) -> dict[str, Path]:
    csv_path, jsonl_path = save_seed_literature()
    dst_dir = run_dir / "literature"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_dir / csv_path.name
    dst_jsonl = dst_dir / jsonl_path.name
    shutil.copy2(csv_path, dst_csv)
    shutil.copy2(jsonl_path, dst_jsonl)
    return {"csv": dst_csv, "jsonl": dst_jsonl}
