from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"无法读取图像：{path}")
    return arr


def imwrite(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    encoded.tofile(str(path))
    return path


def pair_by_stem(image_paths: list[Path], mask_paths: list[Path]) -> dict[Path, Path | None]:
    mask_map = {p.stem: p for p in mask_paths}
    pairs: dict[Path, Path | None] = {}
    for i, img in enumerate(image_paths):
        if img.stem in mask_map:
            pairs[img] = mask_map[img.stem]
        elif i < len(mask_paths):
            pairs[img] = mask_paths[i]
        else:
            pairs[img] = None
    return pairs


def save_pair_table(pairs: dict[Path, Path | None], target: Path) -> Path:
    rows = []
    for img, mask in pairs.items():
        rows.append({
            "image": str(img),
            "image_name": img.name,
            "mask": str(mask) if mask else "",
            "mask_name": mask.name if mask else "",
            "has_mask": mask is not None,
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8-sig")
    return target
