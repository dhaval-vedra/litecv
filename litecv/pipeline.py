"""
litecv.pipeline — public re-export of _pipeline module.

This file exists so that `from litecv.pipeline import enhance_camera`
works as a proper Python sub-module import.
"""
from ._pipeline import (
    Pipeline,
    # filters
    blur, gaussian_blur, median, bilateral, sharpen, unsharp_mask,
    edges, laplacian, emboss, denoise, clahe, equalize,
    # geometry
    resize, scale, crop, roi, pad, rotate, rotate90, flip, transpose,
    # color
    to_gray, to_rgb,
    # adjustments
    brightness, contrast, gamma, invert, normalize,
    threshold, threshold_otsu,
    # effects
    sepia, thermal, night_vision, sketch, cartoon, edge_glow,
    # morphological
    morph, erode, dilate,
    # i/o helpers
    save, save_dir, tap, branch, copy,
    # preset pipelines
    enhance_camera, surveillance, night_mode, document_scan,
)

__all__ = [
    "Pipeline",
    "blur", "gaussian_blur", "median", "bilateral", "sharpen", "unsharp_mask",
    "edges", "laplacian", "emboss", "denoise", "clahe", "equalize",
    "resize", "scale", "crop", "roi", "pad", "rotate", "rotate90", "flip", "transpose",
    "to_gray", "to_rgb",
    "brightness", "contrast", "gamma", "invert", "normalize",
    "threshold", "threshold_otsu",
    "sepia", "thermal", "night_vision", "sketch", "cartoon", "edge_glow",
    "morph", "erode", "dilate",
    "save", "save_dir", "tap", "branch", "copy",
    "enhance_camera", "surveillance", "night_mode", "document_scan",
]
