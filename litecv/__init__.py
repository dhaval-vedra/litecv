"""
litecv-advanced — Lightweight yet powerful computer vision for edge devices
===========================================================================

Zero OpenCV. Zero pygame. Pillow + NumPy only (+ optional C++ extension).
Built for Raspberry Pi, Jetson Nano, and any ARM/x86 edge device.

Quick start:
    import litecv
    img = litecv.open_image("photo.jpg")
    img.clahe().sepia().save("result.jpg")

    # Pipeline style
    from litecv import pipeline as P
    result = img | P.clahe() | P.sepia() | P.save("result.jpg")
"""
from __future__ import annotations
import sys

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "2.0.0"
__author__  = "litecv-advanced"
__license__ = "MIT"

# ── Backend detection ─────────────────────────────────────────────────────────
_EXT_AVAILABLE = False
_EXT_SIMD      = "scalar"

try:
    import _lcv_ext as _ext
    _EXT_AVAILABLE = True
    _EXT_SIMD = _ext.simd_backend()
except ImportError:
    _ext = None

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import numpy as _np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

if not _NP_AVAILABLE:
    raise ImportError("litecv requires numpy. Install with: pip install numpy")

# ── Core exports ──────────────────────────────────────────────────────────────
from ._core import (
    LiteImage,
    FilterType,
    MorphOp,
    MorphShape,
    FlipDir,
)

# Backward compat alias
AdvancedLiteImage = LiteImage

# ── Camera exports ────────────────────────────────────────────────────────────
from ._camera import (
    CameraFeed,
    AsyncCamera,
    FrameBuffer,
)

# ── Pipeline exports ──────────────────────────────────────────────────────────
from . import _pipeline as pipeline
from ._pipeline import Pipeline

# ── Detection exports ─────────────────────────────────────────────────────────
from ._detection import (
    HarrisDetector,
    OpticalFlow,
    MotionDetector,
    TemplateMatcher,
    ContourAnalyzer,
    HOGMatcher,
    KeyPoint,
    FlowVector,
    Detection,
    ContourInfo,
)

# ── Streaming exports ─────────────────────────────────────────────────────────
from ._streaming import MJPEGServer

# ── Profiler exports ──────────────────────────────────────────────────────────
from ._profiler import (
    Profiler,
    fps_counter,
    MemoryMonitor,
    LatencyTracker,
)

# ── ML bridge exports ─────────────────────────────────────────────────────────
from ._ml import (
    TFLiteModel,
    ONNXModel,
    NullModel,
    auto_model,
)

# ── Convenience functions ─────────────────────────────────────────────────────

def open_image(path: str, grayscale: bool = False) -> LiteImage:
    """Load an image from disk."""
    return LiteImage.open(path, grayscale)

def from_numpy(array) -> LiteImage:
    """Create LiteImage from numpy array (RGB or grayscale uint8/float)."""
    import numpy as np
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return LiteImage(arr)

def from_bytes(data: bytes) -> LiteImage:
    """Decode image from JPEG/PNG bytes."""
    return LiteImage.from_bytes(data)

def new_image(width: int, height: int,
              color=(0, 0, 0), channels: int = 3) -> LiteImage:
    """Create a blank image."""
    import numpy as np
    if channels == 1:
        arr = np.full((height, width), color if isinstance(color, int) else color[0],
                      dtype=np.uint8)
    else:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        if isinstance(color, (list, tuple)):
            arr[:, :] = color[:3]
        else:
            arr[:, :] = [color, color, color]
    return LiteImage(arr)

def backend_info() -> dict:
    """Return information about the active backends."""
    info = {
        "version":       __version__,
        "cpp_extension": _EXT_AVAILABLE,
        "simd":          _EXT_SIMD,
        "pillow":        _PIL_AVAILABLE,
        "numpy":         _NP_AVAILABLE,
        "python":        sys.version,
        "platform":      sys.platform,
    }
    if _EXT_AVAILABLE and _ext:
        pool = _ext.pool_stats()
        info["memory_pool"] = pool
    return info

# ── Startup banner ────────────────────────────────────────────────────────────
def _print_banner():
    ext_status = f"C++ ext ✓ [{_EXT_SIMD}]" if _EXT_AVAILABLE else "C++ ext ✗ (pure Python)"
    pil_status = "Pillow ✓" if _PIL_AVAILABLE else "Pillow ✗"
    print(f"[litecv-advanced v{__version__}] {ext_status} | {pil_status}")

_print_banner()

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # Core
    "LiteImage", "AdvancedLiteImage",
    "FilterType", "MorphOp", "MorphShape", "FlipDir",
    # Camera
    "CameraFeed", "AsyncCamera", "FrameBuffer",
    # Pipeline
    "pipeline", "Pipeline",
    # Detection
    "HarrisDetector", "OpticalFlow", "MotionDetector",
    "TemplateMatcher", "ContourAnalyzer", "HOGMatcher",
    "KeyPoint", "FlowVector", "Detection", "ContourInfo",
    # Streaming
    "MJPEGServer",
    # Profiler
    "Profiler", "fps_counter", "MemoryMonitor", "LatencyTracker",
    # ML
    "TFLiteModel", "ONNXModel", "NullModel", "auto_model",
    # Helpers
    "open_image", "from_numpy", "from_bytes", "new_image", "backend_info",
]
