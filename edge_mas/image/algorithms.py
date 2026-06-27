from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, morphology

from edge_mas.config import ExperimentConfig


def ensure_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    arr = gray.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    return arr.astype(np.uint8)


def mask_to_edge_gt(mask: np.ndarray) -> np.ndarray:
    m = ensure_uint8(mask)
    binary = m > 127
    if binary.mean() > 0.5:
        binary = ~binary
    binary = morphology.remove_small_objects(binary, min_size=8)
    eroded = morphology.binary_erosion(binary, morphology.square(3))
    edge = binary ^ eroded
    return (edge.astype(np.uint8) * 255)


def sobel_edge(gray: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    img = ensure_uint8(gray)
    img_blur = cv2.GaussianBlur(img, (0, 0), cfg.sigma)
    sx = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    th = np.percentile(mag, 90)
    return ((mag >= th).astype(np.uint8) * 255)


def prewitt_edge(gray: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    img = ensure_uint8(gray).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), cfg.sigma)
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    th = np.percentile(mag, 90)
    return ((mag >= th).astype(np.uint8) * 255)


def log_edge(gray: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    img = ensure_uint8(gray).astype(np.float32) / 255.0
    smoothed = filters.gaussian(img, sigma=cfg.log_sigma)
    lap = ndi.laplace(smoothed)
    # zero-crossing approximation by local sign changes, filtered by magnitude
    sign = np.sign(lap)
    zc = np.zeros_like(sign, dtype=bool)
    zc[:-1, :] |= sign[:-1, :] * sign[1:, :] < 0
    zc[:, :-1] |= sign[:, :-1] * sign[:, 1:] < 0
    magnitude = np.abs(lap)
    zc &= magnitude > np.percentile(magnitude, 85)
    return zc.astype(np.uint8) * 255


def canny_edge(gray: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    img = ensure_uint8(gray)
    img_blur = cv2.GaussianBlur(img, (0, 0), cfg.sigma)
    edge = cv2.Canny(img_blur, cfg.canny_low, cfg.canny_high)
    return edge


def morph_gradient_edge(gray: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    img = ensure_uint8(gray)
    k = max(3, int(cfg.morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    th = np.percentile(grad, 88)
    return ((grad >= th).astype(np.uint8) * 255)


def run_algorithms(gray: np.ndarray, cfg: ExperimentConfig, selected_algorithms: list[str]) -> dict[str, np.ndarray]:
    funcs = {
        "Sobel": sobel_edge,
        "Prewitt": prewitt_edge,
        "LoG": log_edge,
        "Canny": canny_edge,
        "Morphological Gradient": morph_gradient_edge,
    }
    results: dict[str, np.ndarray] = {}
    for name in selected_algorithms:
        if name not in funcs:
            continue
        results[name] = funcs[name](gray, cfg)
    return results
