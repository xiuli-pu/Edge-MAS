from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(base: Path) -> Path:
    run_dir = base / f"run_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {".", "_", "-", " ", "（", "）", "(", ")"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip() or "file"


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file in uploaded_files or []:
        path = target_dir / safe_filename(file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        paths.append(path)
    return paths


def copy_files(paths: list[Path], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in paths:
        dst = target_dir / safe_filename(p.name)
        shutil.copy2(p, dst)
        copied.append(dst)
    return copied


def zip_directory(src_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(src_dir))
    return zip_path


def read_text_if_exists(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    return ""
