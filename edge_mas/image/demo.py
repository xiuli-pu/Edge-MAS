from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from edge_mas.image.io_utils import imwrite


def generate_demo_dataset(target_dir: Path, n: int = 3, size: tuple[int, int] = (360, 520)) -> tuple[list[Path], list[Path]]:
    img_dir = target_dir / "images"
    mask_dir = target_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    h, w = size
    rng = np.random.default_rng(2026)
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    for i in range(1, n + 1):
        base = np.full((h, w), 178, dtype=np.uint8)
        noise = rng.normal(0, 13, (h, w)).astype(np.int16)
        texture = (12 * np.sin(np.linspace(0, 20, w))[None, :]).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise + texture, 0, 255).astype(np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        x = int(w * (0.15 + 0.05 * i))
        y = int(h * 0.08)
        pts = []
        for t in range(8):
            x += int(rng.integers(20, 55))
            y += int(rng.integers(20, 45))
            x = max(20, min(w - 20, x + int(rng.integers(-50, 50))))
            y = max(20, min(h - 20, y))
            pts.append((x, y))
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(mask, [pts], False, 255, thickness=int(rng.integers(9, 15)))
        cv2.polylines(img, [pts], False, int(45 + 10 * i), thickness=int(rng.integers(3, 6)))

        # add minor distractor texture
        for _ in range(25):
            x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x2, y2 = x1 + int(rng.integers(-30, 30)), y1 + int(rng.integers(-30, 30))
            cv2.line(img, (x1, y1), (x2, y2), int(rng.integers(130, 210)), 1)

        img_path = img_dir / f"demo_crack_{i:02d}.png"
        mask_path = mask_dir / f"demo_crack_{i:02d}.png"
        imwrite(img_path, img)
        imwrite(mask_path, mask)
        image_paths.append(img_path)
        mask_paths.append(mask_path)
    return image_paths, mask_paths
