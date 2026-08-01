"""
litecv._core — LiteImage class
The central image container for litecv-advanced.

Design:
- Stores data as numpy uint8 array (H, W) or (H, W, 3)
- All operations return a NEW LiteImage (immutable-style, safe for pipelines)
- Uses C++ extension when available, falls back to pure Python (Pillow + NumPy)
- No pygame, no OpenCV
"""
from __future__ import annotations

import copy
import math
import os
from enum import Enum, auto
from typing import Optional, Tuple, Union, List

import numpy as np

try:
    from PIL import Image as PILImage, ImageDraw, ImageFilter, ImageEnhance
    _HAS_PILLOW = True
except ImportError:
    _HAS_PILLOW = False

try:
    import _lcv_ext as _ext
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

# ── Enums ─────────────────────────────────────────────────────────────────────

class FilterType(Enum):
    BLUR          = auto()
    SHARPEN       = auto()
    EDGE_DETECT   = auto()
    EMBOSS        = auto()
    DENOISE       = auto()
    MEDIAN        = auto()
    LAPLACIAN     = auto()
    UNSHARP_MASK  = auto()
    BILATERAL     = auto()

class MorphOp(Enum):
    ERODE    = "erode"
    DILATE   = "dilate"
    OPEN     = "open"
    CLOSE    = "close"
    GRADIENT = "gradient"
    TOPHAT   = "tophat"
    BLACKHAT = "blackhat"

class MorphShape(Enum):
    RECT    = "rect"
    ELLIPSE = "ellipse"
    CROSS   = "cross"

class FlipDir(Enum):
    HORIZONTAL = 1
    VERTICAL   = 0
    BOTH       = -1

# ── LiteImage ─────────────────────────────────────────────────────────────────

class LiteImage:
    """
    Edge-optimised image container.

    Channels: always RGB (3) or grayscale (1).
    All operations return a new LiteImage — safe for pipelines and threading.
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("LiteImage requires a numpy array")
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        if data.ndim == 2:
            pass  # grayscale
        elif data.ndim == 3 and data.shape[2] in (1, 3, 4):
            if data.shape[2] == 1:
                data = data[:, :, 0]  # collapse to 2D
            elif data.shape[2] == 4:
                data = data[:, :, :3]  # drop alpha
        else:
            raise ValueError(f"Unsupported array shape: {data.shape}")
        self._data = np.ascontiguousarray(data)

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def channels(self) -> int:
        return 1 if self._data.ndim == 2 else self._data.shape[2]

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def is_gray(self) -> bool:
        return self._data.ndim == 2

    def copy(self) -> "LiteImage":
        return LiteImage(self._data.copy())

    def __repr__(self) -> str:
        return f"LiteImage({self.width}×{self.height}, {self.channels}ch)"

    # ── I/O ───────────────────────────────────────────────────────────────
    @staticmethod
    def open(path: str, grayscale: bool = False) -> "LiteImage":
        if _HAS_EXT:
            arr = _ext.imread(path, grayscale)
            return LiteImage(arr)
        if not _HAS_PILLOW:
            raise RuntimeError("PIL not available — cannot load image")
        mode = "L" if grayscale else "RGB"
        img = PILImage.open(path).convert(mode)
        return LiteImage(np.array(img))

    def save(self, path: str, quality: int = 90) -> "LiteImage":
        if _HAS_EXT:
            _ext.imwrite(path, self._data)
        elif _HAS_PILLOW:
            pil = self._to_pil()
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                pil.save(path, quality=quality)
            else:
                pil.save(path)
        else:
            raise RuntimeError("No backend available for save()")
        return self

    def to_bytes(self, fmt: str = ".jpg") -> bytes:
        if _HAS_EXT:
            return _ext.imencode(self._data, fmt)
        if _HAS_PILLOW:
            import io
            _fmt_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
                        "bmp": "BMP", "webp": "WEBP", "gif": "GIF"}
            pil_fmt = _fmt_map.get(fmt.lstrip(".").lower(), fmt.lstrip(".").upper())
            buf = io.BytesIO()
            self._to_pil().save(buf, format=pil_fmt)
            return buf.getvalue()
        raise RuntimeError("No backend for to_bytes()")

    @staticmethod
    def from_bytes(data: bytes) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.imdecode(data))
        if _HAS_PILLOW:
            import io
            img = PILImage.open(io.BytesIO(data)).convert("RGB")
            return LiteImage(np.array(img))
        raise RuntimeError("No backend for from_bytes()")

    def _to_pil(self) -> "PILImage.Image":
        if not _HAS_PILLOW:
            raise RuntimeError("Pillow not available")
        mode = "L" if self.is_gray else "RGB"
        return PILImage.fromarray(self._data, mode)

    # ── Geometry ──────────────────────────────────────────────────────────
    def resize(self, width: int, height: int) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.resize(self._data, width, height))
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().resize((width, height), PILImage.BILINEAR)))
        # Fallback: nearest-neighbour
        src = self._data
        sy = np.clip((np.arange(height) * self.height / height).astype(int), 0, self.height-1)
        sx = np.clip((np.arange(width)  * self.width  / width ).astype(int), 0, self.width -1)
        return LiteImage(src[np.ix_(sy, sx)])

    def scale(self, fx: float, fy: float = None) -> "LiteImage":
        fy = fy or fx
        return self.resize(int(self.width * fx), int(self.height * fy))

    def crop(self, x: int, y: int, w: int, h: int) -> "LiteImage":
        x2, y2 = min(x+w, self.width), min(y+h, self.height)
        if self.is_gray:
            return LiteImage(self._data[y:y2, x:x2].copy())
        return LiteImage(self._data[y:y2, x:x2, :].copy())

    def roi(self, x: int, y: int, w: int, h: int) -> "LiteImage":
        """Alias for crop — region of interest."""
        return self.crop(x, y, w, h)

    def pad(self, top: int, bottom: int, left: int, right: int,
            value: int = 0) -> "LiteImage":
        if self.is_gray:
            return LiteImage(np.pad(self._data, ((top, bottom), (left, right)),
                                    constant_values=value))
        return LiteImage(np.pad(self._data, ((top, bottom), (left, right), (0, 0)),
                                constant_values=value))

    def rotate(self, degrees: float,
               expand: bool = False) -> "LiteImage":
        """Rotate by arbitrary degrees (counter-clockwise)."""
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().rotate(degrees, expand=expand)))
        # 90° shortcuts via C++
        if _HAS_EXT and degrees in (90, -270):
            return LiteImage(_ext.rotate(self._data, 2))
        if _HAS_EXT and degrees in (180, -180):
            return LiteImage(_ext.rotate(self._data, 1))
        if _HAS_EXT and degrees in (270, -90):
            return LiteImage(_ext.rotate(self._data, 0))
        # Nearest-neighbour arbitrary rotation (slow, use Pillow when possible)
        h, w = self.height, self.width
        angle_rad = math.radians(-degrees)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        cx, cy = w / 2, h / 2
        out = np.zeros_like(self._data)
        for y in range(h):
            for x in range(w):
                sx = int(cos_a*(x-cx) + sin_a*(y-cy) + cx)
                sy = int(-sin_a*(x-cx) + cos_a*(y-cy) + cy)
                if 0 <= sx < w and 0 <= sy < h:
                    out[y, x] = self._data[sy, sx]
        return LiteImage(out)

    def rotate90(self, times: int = 1) -> "LiteImage":
        code = {1: 0, 2: 1, 3: 2}.get(times % 4, 1)
        if _HAS_EXT:
            return LiteImage(_ext.rotate(self._data, code))
        return LiteImage(np.rot90(self._data, k=-(times % 4)))

    def flip(self, direction: Union[FlipDir, str] = FlipDir.HORIZONTAL) -> "LiteImage":
        if isinstance(direction, str):
            direction = FlipDir[direction.upper()]
        if _HAS_EXT:
            return LiteImage(_ext.flip(self._data, direction.value))
        if direction == FlipDir.HORIZONTAL:
            return LiteImage(np.fliplr(self._data).copy())
        elif direction == FlipDir.VERTICAL:
            return LiteImage(np.flipud(self._data).copy())
        else:  # BOTH
            return LiteImage(np.flipud(np.fliplr(self._data)).copy())

    def transpose(self) -> "LiteImage":
        if self.is_gray:
            return LiteImage(self._data.T.copy())
        return LiteImage(self._data.transpose(1, 0, 2).copy())

    # ── Color ─────────────────────────────────────────────────────────────
    def to_gray(self) -> "LiteImage":
        if self.is_gray:
            return self.copy()
        if _HAS_EXT:
            return LiteImage(_ext.to_gray(self._data))
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().convert("L")))
        return LiteImage((0.299*self._data[:,:,0] +
                          0.587*self._data[:,:,1] +
                          0.114*self._data[:,:,2]).astype(np.uint8))

    def to_rgb(self) -> "LiteImage":
        if not self.is_gray:
            return self.copy()
        rgb = np.stack([self._data] * 3, axis=-1)
        return LiteImage(rgb)

    def split_channels(self) -> Tuple["LiteImage", "LiteImage", "LiteImage"]:
        if self.is_gray:
            return (self.copy(), self.copy(), self.copy())
        return (LiteImage(self._data[:,:,0]),
                LiteImage(self._data[:,:,1]),
                LiteImage(self._data[:,:,2]))

    @staticmethod
    def merge_channels(r: "LiteImage", g: "LiteImage", b: "LiteImage") -> "LiteImage":
        return LiteImage(np.stack([r._data, g._data, b._data], axis=-1))

    # ── Pixel adjustments ─────────────────────────────────────────────────
    def brightness(self, delta: int) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.brightness(self._data, delta))
        return LiteImage(np.clip(self._data.astype(int) + delta, 0, 255).astype(np.uint8))

    def contrast(self, scale: float) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.contrast(self._data, scale))
        arr = ((self._data.astype(float) - 128) * scale + 128).clip(0, 255)
        return LiteImage(arr.astype(np.uint8))

    def gamma(self, gamma_val: float) -> "LiteImage":
        lut = np.array([min(255, int((i/255.0) ** (1.0/gamma_val) * 255))
                        for i in range(256)], dtype=np.uint8)
        return LiteImage(lut[self._data])

    def invert(self) -> "LiteImage":
        return LiteImage(255 - self._data)

    def normalize(self) -> "LiteImage":
        mn, mx = self._data.min(), self._data.max()
        if mx == mn:
            return self.copy()
        return LiteImage(((self._data - mn) * 255.0 / (mx - mn)).astype(np.uint8))

    def threshold(self, thresh: int = 128, maxval: int = 255,
                  invert: bool = False) -> "LiteImage":
        gray = self.to_gray()._data
        mask = gray > thresh
        if invert:
            mask = ~mask
        return LiteImage(np.where(mask, maxval, 0).astype(np.uint8))

    def threshold_otsu(self) -> "LiteImage":
        gray = self.to_gray()._data.ravel()
        hist = np.bincount(gray, minlength=256).astype(float)
        total = gray.size
        best_t, best_var = 0, 0.0
        w0, sum0 = 0, 0
        total_sum = np.sum(np.arange(256) * hist)
        for t in range(256):
            w0 += hist[t]; w1 = total - w0
            if w0 == 0 or w1 == 0: continue
            sum0 += t * hist[t]
            m0, m1 = sum0/w0, (total_sum - sum0)/w1
            var = w0 * w1 * (m0 - m1)**2
            if var > best_var:
                best_var, best_t = var, t
        return self.threshold(int(best_t))

    def absdiff(self, other: "LiteImage") -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.absdiff(self._data, other._data))
        return LiteImage(np.abs(self._data.astype(int) - other._data.astype(int)).astype(np.uint8))

    # ── Filters ───────────────────────────────────────────────────────────
    def filter(self, filter_type: Union[FilterType, str],
               kernel_size: int = 3, **kwargs) -> "LiteImage":
        if isinstance(filter_type, str):
            filter_type = FilterType[filter_type.upper()]

        if filter_type == FilterType.BLUR:
            return self.blur(kernel_size)
        elif filter_type == FilterType.SHARPEN:
            return self.sharpen()
        elif filter_type == FilterType.EDGE_DETECT:
            return self.edges()
        elif filter_type == FilterType.EMBOSS:
            return self.emboss()
        elif filter_type == FilterType.DENOISE:
            return self.denoise()
        elif filter_type == FilterType.MEDIAN:
            return self.median(kernel_size)
        elif filter_type == FilterType.LAPLACIAN:
            return self.laplacian()
        elif filter_type == FilterType.UNSHARP_MASK:
            return self.unsharp_mask(**kwargs)
        elif filter_type == FilterType.BILATERAL:
            return self.bilateral(**kwargs)
        return self.copy()

    def blur(self, radius: int = 1) -> "LiteImage":
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(
                ImageFilter.GaussianBlur(radius=radius))))
        # Box blur fallback
        from scipy.ndimage import uniform_filter
        try:
            return LiteImage(uniform_filter(self._data, size=2*radius+1))
        except ImportError:
            return self.copy()

    def gaussian_blur(self, sigma: float = 1.0) -> "LiteImage":
        return self.blur(int(sigma))

    def median(self, ksize: int = 3) -> "LiteImage":
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(
                ImageFilter.MedianFilter(size=ksize))))
        return self.copy()

    def bilateral(self, d: int = 9,
                  sigma_color: float = 75.0,
                  sigma_space: float = 75.0) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.bilateral(self._data, d, sigma_color, sigma_space))
        return self.blur(d // 2)  # Pillow fallback

    def sharpen(self) -> "LiteImage":
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(ImageFilter.SHARPEN)))
        return self.copy()

    def unsharp_mask(self, amount: float = 1.5, radius: int = 3) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.unsharp(self._data, amount, radius))
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(
                ImageFilter.UnsharpMask(radius=radius, percent=int(amount*100)))))
        return self.copy()

    def edges(self) -> "LiteImage":
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(ImageFilter.FIND_EDGES)))
        gray = self.to_gray()._data.astype(float)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        return LiteImage(np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8))

    def laplacian(self) -> "LiteImage":
        # Pure-numpy Laplacian (no PIL Kernel size issue across versions)
        gray = self.to_gray()._data.astype(float)
        k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(gray, 1, mode='edge')
        windows = sliding_window_view(pad, (3, 3))
        result = np.clip((windows * k).sum(axis=(-2,-1)) + 128, 0, 255).astype(np.uint8)
        return LiteImage(result)

    def emboss(self) -> "LiteImage":
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(ImageFilter.EMBOSS)))
        return self.copy()

    def denoise(self) -> "LiteImage":
        if _HAS_PILLOW:
            return LiteImage(np.array(self._to_pil().filter(ImageFilter.SMOOTH_MORE)))
        return self.copy()

    def custom_kernel(self, kernel: list, scale: float = 1.0,
                       offset: int = 0) -> "LiteImage":
        size = int(len(kernel) ** 0.5)
        if _HAS_PILLOW:
            k = [int(v / scale) for v in kernel]
            return LiteImage(np.array(self._to_pil().filter(
                ImageFilter.Kernel(size=size, kernel=k, scale=1, offset=offset))))
        return self.copy()

    # ── Advanced filters ──────────────────────────────────────────────────
    def clahe(self, tile_w: int = 8, tile_h: int = 8,
              clip_limit: float = 2.0) -> "LiteImage":
        """Contrast Limited Adaptive Histogram Equalization (great for low-light cameras)."""
        if _HAS_EXT:
            return LiteImage(_ext.clahe(self._data, tile_w, tile_h, clip_limit))
        # Pillow fallback: global equalization
        if _HAS_PILLOW and _HAS_EXT is False:
            from PIL import ImageOps
            return LiteImage(np.array(ImageOps.equalize(self._to_pil())))
        return self.copy()

    def equalize(self) -> "LiteImage":
        """Global histogram equalization."""
        if _HAS_EXT and self.is_gray:
            return LiteImage(_ext.equalize(self._data))
        if _HAS_PILLOW:
            from PIL import ImageOps
            return LiteImage(np.array(ImageOps.equalize(self._to_pil())))
        return self.copy()

    # ── Morphological operations ──────────────────────────────────────────
    def morph(self, op: Union[MorphOp, str] = MorphOp.ERODE,
              ksize: int = 3,
              shape: Union[MorphShape, str] = MorphShape.RECT) -> "LiteImage":
        """Morphological operation on grayscale image."""
        if isinstance(op, str):    op    = MorphOp[op.upper()]
        if isinstance(shape, str): shape = MorphShape[shape.upper()]
        if _HAS_EXT:
            gray = self.to_gray()._data
            result = _ext.morph(gray, op.value, ksize, shape.value)
            return LiteImage(result)
        # Pillow fallback (only erode/dilate via min/max filter)
        if _HAS_PILLOW:
            from PIL import ImageFilter
            gray_pil = self.to_gray()._to_pil()
            if op in (MorphOp.ERODE, MorphOp.OPEN):
                r = LiteImage(np.array(gray_pil.filter(ImageFilter.MinFilter(ksize))))
            else:
                r = LiteImage(np.array(gray_pil.filter(ImageFilter.MaxFilter(ksize))))
            return r
        return self.copy()

    def erode(self, ksize: int = 3) -> "LiteImage":
        return self.morph(MorphOp.ERODE, ksize)

    def dilate(self, ksize: int = 3) -> "LiteImage":
        return self.morph(MorphOp.DILATE, ksize)

    # ── Effects (C++ or Pillow fallback) ──────────────────────────────────
    def sepia(self) -> "LiteImage":
        if _HAS_EXT and not self.is_gray:
            return LiteImage(_ext.effect_sepia(self._data))
        # Pillow/numpy fallback
        rgb = self.to_rgb()._data.astype(float)
        r = np.clip(rgb[:,:,0]*0.393 + rgb[:,:,1]*0.769 + rgb[:,:,2]*0.189, 0, 255)
        g = np.clip(rgb[:,:,0]*0.349 + rgb[:,:,1]*0.686 + rgb[:,:,2]*0.168, 0, 255)
        b = np.clip(rgb[:,:,0]*0.272 + rgb[:,:,1]*0.534 + rgb[:,:,2]*0.131, 0, 255)
        return LiteImage(np.stack([r, g, b], axis=-1).astype(np.uint8))

    def thermal(self) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.effect_thermal(self.to_rgb()._data))
        gray = self.to_gray()._data.astype(float) / 255.0
        r = np.clip(255 * np.minimum(1, 3 * gray), 0, 255)
        g = np.clip(255 * np.maximum(0, np.minimum(1, 3*gray - 1)), 0, 255)
        b = np.clip(255 * np.maximum(0, np.minimum(1, 3*gray - 2)), 0, 255)
        return LiteImage(np.stack([r, g, b], axis=-1).astype(np.uint8))

    def night_vision(self) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.effect_night_vision(self.to_rgb()._data))
        enhanced = self.to_gray().clahe()._data
        out = np.zeros((*enhanced.shape, 3), dtype=np.uint8)
        out[:, :, 1] = enhanced
        return LiteImage(out)

    def sketch(self) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.effect_sketch(self.to_rgb()._data))
        if _HAS_PILLOW:
            edges = np.array(self._to_pil().convert("L").filter(ImageFilter.FIND_EDGES))
            inv = 255 - edges
            return LiteImage(np.stack([inv, inv, inv], axis=-1))
        return self.copy()

    def cartoon(self) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.effect_cartoon(self.to_rgb()._data))
        # Numpy fallback: posterize + edge mask
        rgb = self.to_rgb()._data
        quantized = ((rgb // 64) * 64 + 32).astype(np.uint8)
        if _HAS_PILLOW:
            edges = np.array(self._to_pil().convert("L").filter(ImageFilter.FIND_EDGES))
            mask = (edges < 30).astype(np.uint8)[:, :, np.newaxis]
            return LiteImage(quantized * mask)
        return LiteImage(quantized)

    def edge_glow(self, color: Tuple[int,int,int] = (0, 255, 120)) -> "LiteImage":
        if _HAS_EXT:
            return LiteImage(_ext.effect_edge_glow(
                self.to_rgb()._data, color[0], color[1], color[2]))
        edges = self.to_gray().edges()._data.astype(float) / 255.0
        r = (color[0] * edges).astype(np.uint8)
        g = (color[1] * edges).astype(np.uint8)
        b = (color[2] * edges).astype(np.uint8)
        return LiteImage(np.stack([r, g, b], axis=-1))

    # ── Drawing ───────────────────────────────────────────────────────────
    def draw_rect(self, x: int, y: int, w: int, h: int,
                  color=(255,0,0), thickness: int = 2) -> "LiteImage":
        img = self.copy()
        if _HAS_PILLOW:
            pil = img._to_pil()
            draw = ImageDraw.Draw(pil)
            c = color if not img.is_gray else color[0]
            for i in range(thickness):
                draw.rectangle([x+i, y+i, x+w-i, y+h-i], outline=c)
            return LiteImage(np.array(pil))
        d = img._data
        for i in range(thickness):
            if not img.is_gray:
                d[max(0,y+i):min(d.shape[0],y+h-i+1), max(0,x+i), :] = color
                d[max(0,y+i):min(d.shape[0],y+h-i+1), min(d.shape[1]-1,x+w-i), :] = color
                d[max(0,y+i), max(0,x+i):min(d.shape[1],x+w-i+1), :] = color
                d[min(d.shape[0]-1,y+h-i), max(0,x+i):min(d.shape[1],x+w-i+1), :] = color
        return img

    def draw_circle(self, cx: int, cy: int, radius: int,
                    color=(0,255,0), thickness: int = 2) -> "LiteImage":
        img = self.copy()
        if _HAS_PILLOW:
            pil = img._to_pil()
            draw = ImageDraw.Draw(pil)
            c = color if not img.is_gray else color[0]
            for i in range(thickness):
                r = radius - i
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=c)
            return LiteImage(np.array(pil))
        return img

    def draw_line(self, x1: int, y1: int, x2: int, y2: int,
                  color=(255,255,0), thickness: int = 2) -> "LiteImage":
        img = self.copy()
        if _HAS_PILLOW:
            pil = img._to_pil()
            draw = ImageDraw.Draw(pil)
            c = color if not img.is_gray else color[0]
            draw.line([x1, y1, x2, y2], fill=c, width=thickness)
            return LiteImage(np.array(pil))
        return img

    def draw_text(self, text: str, x: int, y: int,
                  color=(255,255,255), font_size: int = 16) -> "LiteImage":
        img = self.copy()
        if _HAS_PILLOW:
            pil = img._to_pil()
            draw = ImageDraw.Draw(pil)
            c = color if not img.is_gray else color[0]
            draw.text((x, y), text, fill=c)
            return LiteImage(np.array(pil))
        return img

    def draw_keypoints(self, keypoints: list,
                       color=(0, 255, 0), radius: int = 4) -> "LiteImage":
        """Draw Harris corners or optical flow points. keypoints = [(x, y, ...)]."""
        img = self.copy()
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            img = img.draw_circle(x, y, radius, color, 2)
        return img

    def draw_contours(self, contours: list, color=(0, 255, 0)) -> "LiteImage":
        """Draw contour points. contours = list of contour dicts (from find_contours)."""
        img = self.copy()
        if _HAS_PILLOW:
            pil = img._to_pil()
            draw = ImageDraw.Draw(pil)
            c = color if not img.is_gray else color[0]
            for contour in contours:
                pts = contour.get("points", [])
                for pt in pts:
                    x, y = int(pt[0]), int(pt[1])
                    draw.point((x, y), fill=c)
            return LiteImage(np.array(pil))
        return img

    # ── Analysis ──────────────────────────────────────────────────────────
    def histogram(self) -> list:
        if _HAS_EXT:
            return _ext.histogram(self.to_gray()._data)
        return np.bincount(self.to_gray()._data.ravel(), minlength=256).tolist()

    def dominant_colors(self, num_colors: int = 5) -> list:
        if _HAS_EXT and not self.is_gray:
            return _ext.dominant_colors(self._data, num_colors)
        # Numpy fallback: quantize
        rgb = self.to_rgb()._data
        q = 32
        bins = (rgb // q).reshape(-1, 3)
        unique, counts = np.unique(bins, axis=0, return_counts=True)
        order = np.argsort(-counts)[:num_colors]
        total = rgb.shape[0] * rgb.shape[1]
        result = []
        for i in order:
            r, g, b2 = (unique[i] * q + q//2).clip(0, 255)
            result.append({
                "r": int(r), "g": int(g), "b": int(b2),
                "fraction": float(counts[i]) / total,
                "hex": f"#{int(r):02X}{int(g):02X}{int(b2):02X}"
            })
        return result

    def mean_color(self) -> Tuple[float, ...]:
        if self.is_gray:
            return (float(self._data.mean()),)
        return tuple(float(self._data[:,:,c].mean()) for c in range(3))

    def std_color(self) -> Tuple[float, ...]:
        if self.is_gray:
            return (float(self._data.std()),)
        return tuple(float(self._data[:,:,c].std()) for c in range(3))

    # ── Pipeline operator ─────────────────────────────────────────────────
    def __or__(self, func):
        """img | operation syntax for pipelines."""
        return func(self)

    def __ior__(self, func):
        return func(self)
