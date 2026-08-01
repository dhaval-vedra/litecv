"""
litecv._pipeline — Chainable pipeline API for image processing

Two ways to use:
  1. Operator chaining:  img | blur(3) | sepia | save("out.jpg")
  2. Pipeline object:    Pipeline([blur(3), sepia, save("out.jpg")])(img)

All operations are lazy — nothing runs until you call the pipeline.
Pipelines are reusable and thread-safe (they don't store state).

Examples:
    from litecv import open_image
    from litecv.pipeline import blur, sepia, clahe, save, resize, edges, thermal

    # Operator style
    result = open_image("photo.jpg") | clahe() | sepia | save("sepia.jpg")

    # Pipeline object style (reuse across many images)
    enhance = Pipeline([clahe(clip_limit=3.0), unsharp_mask(amount=2.0), save_dir("output/")])
    for path in image_paths:
        enhance(open_image(path))
"""
from __future__ import annotations

from typing import Callable, List, Optional, Any
from pathlib import Path

from ._core import LiteImage, FilterType, MorphOp, MorphShape, FlipDir


# ── Step factory functions ────────────────────────────────────────────────────
# Each returns a callable (LiteImage → LiteImage)

def blur(radius: int = 1):
    """Gaussian blur."""
    return lambda img: img.blur(radius)

def gaussian_blur(sigma: float = 1.0):
    return lambda img: img.gaussian_blur(sigma)

def median(ksize: int = 3):
    return lambda img: img.median(ksize)

def bilateral(d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0):
    """Edge-preserving bilateral filter."""
    return lambda img: img.bilateral(d, sigma_color, sigma_space)

def sharpen():
    return lambda img: img.sharpen()

def unsharp_mask(amount: float = 1.5, radius: int = 3):
    return lambda img: img.unsharp_mask(amount, radius)

def edges():
    """Sobel edge detection."""
    return lambda img: img.edges()

def laplacian():
    return lambda img: img.laplacian()

def emboss():
    return lambda img: img.emboss()

def denoise():
    return lambda img: img.denoise()

def clahe(tile_w: int = 8, tile_h: int = 8, clip_limit: float = 2.0):
    """CLAHE — great for low-light cameras."""
    return lambda img: img.clahe(tile_w, tile_h, clip_limit)

def equalize():
    return lambda img: img.equalize()

def resize(width: int, height: int):
    return lambda img: img.resize(width, height)

def scale(fx: float, fy: float = None):
    return lambda img: img.scale(fx, fy or fx)

def crop(x: int, y: int, w: int, h: int):
    return lambda img: img.crop(x, y, w, h)

def roi(x: int, y: int, w: int, h: int):
    """Region of interest — same as crop."""
    return crop(x, y, w, h)

def pad(top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, value: int = 0):
    return lambda img: img.pad(top, bottom, left, right, value)

def rotate(degrees: float, expand: bool = False):
    return lambda img: img.rotate(degrees, expand)

def rotate90(times: int = 1):
    return lambda img: img.rotate90(times)

def flip(direction=FlipDir.HORIZONTAL):
    return lambda img: img.flip(direction)

def transpose():
    return lambda img: img.transpose()

def to_gray():
    return lambda img: img.to_gray()

def to_rgb():
    return lambda img: img.to_rgb()

def brightness(delta: int):
    return lambda img: img.brightness(delta)

def contrast(scale: float):
    return lambda img: img.contrast(scale)

def gamma(value: float):
    return lambda img: img.gamma(value)

def invert():
    return lambda img: img.invert()

def normalize():
    return lambda img: img.normalize()

def threshold(thresh: int = 128, maxval: int = 255, invert_: bool = False):
    return lambda img: img.threshold(thresh, maxval, invert_)

def threshold_otsu():
    return lambda img: img.threshold_otsu()

# Effects
def sepia():
    return lambda img: img.sepia()

def thermal():
    return lambda img: img.thermal()

def night_vision():
    return lambda img: img.night_vision()

def sketch():
    return lambda img: img.sketch()

def cartoon():
    return lambda img: img.cartoon()

def edge_glow(color=(0, 255, 120)):
    return lambda img: img.edge_glow(color)

# Morphological
def morph(op=MorphOp.ERODE, ksize: int = 3, shape=MorphShape.RECT):
    return lambda img: img.morph(op, ksize, shape)

def erode(ksize: int = 3):
    return lambda img: img.erode(ksize)

def dilate(ksize: int = 3):
    return lambda img: img.dilate(ksize)

# Save operations
def save(path: str, quality: int = 90):
    """Save and pass image through (non-terminal)."""
    def _save(img: LiteImage) -> LiteImage:
        img.save(path, quality)
        return img
    return _save

def save_dir(directory: str, prefix: str = "frame_", ext: str = ".jpg"):
    """Save each image to a directory with auto-incrementing filename."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    counter = [0]
    def _save(img: LiteImage) -> LiteImage:
        path = str(Path(directory) / f"{prefix}{counter[0]:06d}{ext}")
        img.save(path)
        counter[0] += 1
        return img
    return _save

def tap(func: Callable[[LiteImage], Any]):
    """Call func(img) for side effects (logging, display), pass img through."""
    def _tap(img: LiteImage) -> LiteImage:
        func(img)
        return img
    return _tap

def branch(*pipelines):
    """Fork: run multiple pipeline steps, return original image."""
    def _branch(img: LiteImage) -> LiteImage:
        for p in pipelines:
            p(img)
        return img
    return _branch

def copy():
    """Make a copy of the image before continuing."""
    return lambda img: img.copy()


# ── Pipeline class ────────────────────────────────────────────────────────────

class Pipeline:
    """
    Composable, reusable image processing pipeline.

    Usage:
        p = Pipeline([clahe(), unsharp_mask(), sepia()])
        result = p(image)

        # Or apply to a list of images
        results = p.map(images)

        # Compose pipelines
        full = p1 >> p2  # run p1 then p2
    """

    def __init__(self, steps: List[Callable] = None):
        self._steps: List[Callable] = list(steps or [])

    def add(self, step: Callable) -> "Pipeline":
        """Add a step (returns new Pipeline, original unchanged)."""
        return Pipeline(self._steps + [step])

    def __call__(self, img: LiteImage) -> LiteImage:
        """Run the pipeline on one image."""
        result = img
        for step in self._steps:
            result = step(result)
        return result

    def map(self, images) -> list:
        """Apply pipeline to an iterable of images."""
        return [self(img) for img in images]

    def __rshift__(self, other: "Pipeline") -> "Pipeline":
        """Compose: pipeline1 >> pipeline2."""
        if isinstance(other, Pipeline):
            return Pipeline(self._steps + other._steps)
        return Pipeline(self._steps + [other])

    def __rrshift__(self, img: LiteImage) -> LiteImage:
        return self(img)

    def benchmark(self, img: LiteImage, n: int = 10) -> dict:
        """Benchmark this pipeline on one image, return timing stats."""
        import time
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self(img)
            times.append(time.perf_counter() - t0)
        return {
            "n": n,
            "mean_ms": sum(times) / n * 1000,
            "min_ms":  min(times) * 1000,
            "max_ms":  max(times) * 1000,
            "fps":     1.0 / (sum(times) / n),
        }


# ── Convenience: named preset pipelines ──────────────────────────────────────

def enhance_camera() -> Pipeline:
    """Preset for raw camera frames: CLAHE → denoise → unsharp."""
    return Pipeline([clahe(clip_limit=2.5), denoise(), unsharp_mask(amount=1.2)])

def surveillance() -> Pipeline:
    """Preset for surveillance: denoise → sharpen → equalize."""
    return Pipeline([denoise(), sharpen(), equalize()])

def night_mode() -> Pipeline:
    """Preset for dark/night scenes: CLAHE → night_vision → unsharp."""
    return Pipeline([clahe(clip_limit=4.0), night_vision(), unsharp_mask()])

def document_scan() -> Pipeline:
    """Preset for document scanning: grayscale → CLAHE → threshold."""
    return Pipeline([to_gray(), clahe(clip_limit=3.0), threshold_otsu()])
