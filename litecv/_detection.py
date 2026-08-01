"""
litecv._detection — Computer vision detection primitives

All work on LiteImage. Returns structured results, no OpenCV needed.

Features:
  - HarrisDetector  — corner / feature point detection
  - OpticalFlow     — Lucas-Kanade sparse flow + dense approximation
  - MotionDetector  — background subtraction + contour finding
  - ObjectDetector  — HOG sliding window (grayscale feature matching)
  - TemplateMatcher — find a template in an image
  - ContourAnalyzer — analyze shapes in binary masks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

try:
    import _lcv_ext as _ext
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

from ._core import LiteImage


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KeyPoint:
    x: int
    y: int
    score: float = 0.0

    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class FlowVector:
    x: float; y: float      # starting point
    u: float; v: float      # displacement (dx, dy)
    valid: bool = True

    @property
    def magnitude(self) -> float:
        return (self.u**2 + self.v**2) ** 0.5

    @property
    def direction_deg(self) -> float:
        import math
        return math.degrees(math.atan2(self.v, self.u))

    @property
    def end_point(self) -> Tuple[int, int]:
        return (int(self.x + self.u), int(self.y + self.v))


@dataclass
class Detection:
    x: int; y: int; w: int; h: int
    score: float = 0.0
    label: str = ""

    @property
    def cx(self) -> int: return self.x + self.w // 2

    @property
    def cy(self) -> int: return self.y + self.h // 2

    @property
    def area(self) -> int: return self.w * self.h

    def iou(self, other: "Detection") -> float:
        """Intersection over Union."""
        ix = max(self.x, other.x); iy = max(self.y, other.y)
        ix2 = min(self.x+self.w, other.x+other.w)
        iy2 = min(self.y+self.h, other.y+other.h)
        if ix2 <= ix or iy2 <= iy: return 0.0
        inter = (ix2-ix) * (iy2-iy)
        return inter / (self.area + other.area - inter)


@dataclass
class ContourInfo:
    points: List[Tuple[int,int]]
    bbox: Tuple[int,int,int,int]  # x, y, w, h
    area: float
    length: int


# ── Harris Corner Detector ────────────────────────────────────────────────────

class HarrisDetector:
    """
    Detect corners / feature points using Harris corner response.

    Usage:
        detector = HarrisDetector(max_corners=200, threshold=0.02)
        keypoints = detector.detect(image)
        annotated = image.draw_keypoints([(kp.x, kp.y) for kp in keypoints])
    """

    def __init__(self, max_corners: int = 500, threshold: float = 0.01,
                 block_size: int = 3, k: float = 0.04):
        self.max_corners = max_corners
        self.threshold = threshold
        self.block_size = block_size
        self.k = k

    def detect(self, img: LiteImage) -> List[KeyPoint]:
        gray = img.to_gray()
        if _HAS_EXT:
            raw = _ext.harris_corners(
                gray._data, self.block_size, 3, self.k,
                self.threshold, self.max_corners)
            return [KeyPoint(x, y, score) for x, y, score in raw]
        return self._detect_python(gray)

    @staticmethod
    def _box_filter(arr: np.ndarray, size: int) -> np.ndarray:
        """Pure-numpy box (uniform) filter — no scipy needed."""
        from numpy.lib.stride_tricks import sliding_window_view
        pad = size // 2
        padded = np.pad(arr, pad, mode='edge')
        wins = sliding_window_view(padded, (size, size))
        return wins.mean(axis=(-2, -1))

    def _detect_python(self, gray: LiteImage) -> List[KeyPoint]:
        """Pure-numpy Harris implementation."""
        img = gray._data.astype(float)
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        A = self._box_filter(gx*gx, self.block_size)
        B = self._box_filter(gy*gy, self.block_size)
        C = self._box_filter(gx*gy, self.block_size)
        R = A*B - C*C - self.k*(A+B)**2
        maxR = R.max()
        if maxR <= 0:
            return []
        R_norm = R / maxR
        corners = []
        h, w = img.shape
        for y in range(5, h-5):
            for x in range(5, w-5):
                if R_norm[y,x] < self.threshold:
                    continue
                local = R_norm[max(0,y-5):y+6, max(0,x-5):x+6]
                if R_norm[y,x] == local.max():
                    corners.append(KeyPoint(x, y, float(R_norm[y,x])))
        corners.sort(key=lambda c: -c.score)
        return corners[:self.max_corners]


# ── Optical Flow ──────────────────────────────────────────────────────────────

class OpticalFlow:
    """
    Lucas-Kanade sparse optical flow for tracking feature points.

    Usage:
        flow = OpticalFlow(win_size=15)
        # First frame: detect keypoints
        kps = HarrisDetector(max_corners=100).detect(frame1)
        pts = [(kp.x, kp.y) for kp in kps]
        # Next frame: track them
        vectors = flow.calc(frame1, frame2, pts)
        for v in vectors:
            if v.valid:
                print(f"Point ({v.x},{v.y}) moved by ({v.u:.1f},{v.v:.1f})")
    """

    def __init__(self, win_size: int = 15, max_iter: int = 30, eps: float = 0.01):
        self.win_size = win_size
        self.max_iter = max_iter
        self.eps = eps

    def calc(self, prev: LiteImage, next_frame: LiteImage,
             points: List[Tuple[float, float]]) -> List[FlowVector]:
        prev_g = prev.to_gray()
        next_g = next_frame.to_gray()
        if _HAS_EXT and points:
            raw = _ext.lk_optical_flow(
                prev_g._data, next_g._data, points,
                self.win_size, self.max_iter, self.eps)
            return [FlowVector(x, y, u, v, valid) for x, y, u, v, valid in raw]
        return self._calc_python(prev_g, next_g, points)

    def _calc_python(self, prev_g, next_g, points):
        """Simplified optical flow fallback (frame difference direction)."""
        diff = next_g._data.astype(int) - prev_g._data.astype(int)
        gy = np.gradient(diff.astype(float), axis=0)
        gx = np.gradient(diff.astype(float), axis=1)
        results = []
        for (px, py) in points:
            x, y = int(px), int(py)
            h, w = diff.shape
            if 0 <= x < w and 0 <= y < h:
                u = float(-gx[y, x]) * 0.1
                v = float(-gy[y, x]) * 0.1
                results.append(FlowVector(float(px), float(py), u, v, True))
            else:
                results.append(FlowVector(float(px), float(py), 0, 0, False))
        return results

    def draw_flow(self, img: LiteImage, vectors: List[FlowVector],
                  color=(0, 255, 0), scale: float = 3.0) -> LiteImage:
        """Draw flow vectors on image."""
        result = img.copy()
        for v in vectors:
            if not v.valid:
                continue
            x1, y1 = int(v.x), int(v.y)
            x2, y2 = int(v.x + v.u * scale), int(v.y + v.v * scale)
            result = result.draw_line(x1, y1, x2, y2, color, 1)
            result = result.draw_circle(x1, y1, 3, color, -1)
        return result


# ── Motion Detector ───────────────────────────────────────────────────────────

class MotionDetector:
    """
    Detect motion using background subtraction + contour analysis.

    Uses the C++ running-average BackgroundSubtractor when available.
    Falls back to frame differencing.

    Usage:
        detector = MotionDetector(threshold=30, min_area=500)
        with CameraFeed("/dev/video0") as cam:
            for frame in cam:
                detections = detector.detect(frame)
                for d in detections:
                    frame = frame.draw_rect(d.x, d.y, d.w, d.h)
    """

    def __init__(self, threshold: int = 25, min_area: int = 500,
                 learning_rate: float = 0.05,
                 morph_ksize: int = 5):
        self.threshold = threshold
        self.min_area = min_area
        self.morph_ksize = morph_ksize
        self._bg_sub = None
        self._prev_frame: Optional[LiteImage] = None

        if _HAS_EXT:
            self._bg_sub = _ext.BackgroundSubtractor(learning_rate, threshold)

    def detect(self, frame: LiteImage) -> List[Detection]:
        gray = frame.to_gray()
        mask = self._get_mask(gray)
        if mask is None:
            return []
        # Morphological cleanup
        cleaned = mask.erode(self.morph_ksize).dilate(self.morph_ksize)
        # Find contours
        return self._contours_to_detections(cleaned)

    def _get_mask(self, gray: LiteImage) -> Optional[LiteImage]:
        if self._bg_sub is not None:
            raw_mask = self._bg_sub.apply(gray._data)
            return LiteImage(raw_mask)
        # Frame difference fallback
        if self._prev_frame is None:
            self._prev_frame = gray
            return None
        diff = gray.absdiff(self._prev_frame).threshold(self.threshold)
        self._prev_frame = gray
        return diff

    def _contours_to_detections(self, mask: LiteImage) -> List[Detection]:
        if _HAS_EXT:
            contours = _ext.find_contours(mask._data, min_len=10)
            result = []
            for c in contours:
                x, y, w, h = c["bbox"]
                area = c["area"]
                if area >= self.min_area:
                    result.append(Detection(x, y, w, h, score=float(area), label="motion"))
            return result
        # Numpy fallback: connected components approximation
        binary = mask._data > 128
        if not binary.any():
            return []
        # Simple row/col projection to find bounding box
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return []
        x, y = int(cols[0]), int(rows[0])
        w, h = int(cols[-1]-cols[0]+1), int(rows[-1]-rows[0]+1)
        area = int(binary.sum())
        if area >= self.min_area:
            return [Detection(x, y, w, h, score=float(area), label="motion")]
        return []

    def reset(self):
        if self._bg_sub:
            self._bg_sub.reset()
        self._prev_frame = None


# ── Template Matcher ──────────────────────────────────────────────────────────

class TemplateMatcher:
    """
    Find a template image within a larger image.

    Works on grayscale. NCC (Normalized Cross Correlation) is more
    robust to lighting changes than SSD.

    Usage:
        matcher = TemplateMatcher("template.jpg")
        result = matcher.match(scene)
        print(f"Found at ({result.x}, {result.y}) score={result.score:.3f}")
        scene = scene.draw_rect(result.x, result.y,
                                matcher.template.width, matcher.template.height)
    """

    def __init__(self, template: "str | LiteImage",
                 method: str = "ncc",
                 threshold: float = 0.7):
        if isinstance(template, str):
            from ._core import LiteImage as LI
            template = LI.open(template)
        self.template = template.to_gray()
        self.method = method
        self.threshold = threshold

    def match(self, scene: LiteImage) -> Optional[Detection]:
        gray_scene = scene.to_gray()
        if _HAS_EXT:
            x, y, score = _ext.template_match(
                gray_scene._data, self.template._data, self.method)
            if self.method == "ncc" and score < self.threshold:
                return None
            return Detection(x, y, self.template.width, self.template.height,
                             score=score, label="template")
        return self._match_python(gray_scene)

    def _match_python(self, gray_scene: LiteImage) -> Optional[Detection]:
        """Pure-numpy NCC template matching."""
        img = gray_scene._data.astype(float)
        tmpl = self.template._data.astype(float)
        th, tw = tmpl.shape
        ih, iw = img.shape
        if th > ih or tw > iw:
            return None
        t_mean = tmpl.mean()
        t_centered = tmpl - t_mean
        t_std = np.sqrt((t_centered**2).sum())
        if t_std < 1e-10:
            return None
        best_x, best_y, best_score = 0, 0, -1.0
        for y in range(ih - th + 1):
            for x in range(iw - tw + 1):
                region = img[y:y+th, x:x+tw]
                r_centered = region - region.mean()
                r_std = np.sqrt((r_centered**2).sum())
                if r_std < 1e-10:
                    continue
                score = float((r_centered * t_centered).sum() / (r_std * t_std))
                if score > best_score:
                    best_score, best_x, best_y = score, x, y
        if best_score < self.threshold:
            return None
        return Detection(best_x, best_y, tw, th, score=best_score, label="template")


# ── Contour Analyzer ──────────────────────────────────────────────────────────

class ContourAnalyzer:
    """
    Find and analyze shapes in a binary mask.

    Usage:
        mask = image.to_gray().threshold_otsu()
        analyzer = ContourAnalyzer(min_area=100)
        contours = analyzer.find(mask)
        for c in contours:
            print(f"  Shape at {c.bbox}, area={c.area:.0f}")
    """

    def __init__(self, min_area: float = 100.0, min_length: int = 10):
        self.min_area = min_area
        self.min_length = min_length

    def find(self, mask: LiteImage) -> List[ContourInfo]:
        gray = mask.to_gray()
        if _HAS_EXT:
            raw = _ext.find_contours(gray._data, self.min_length)
            result = []
            for c in raw:
                if c["area"] >= self.min_area:
                    result.append(ContourInfo(
                        points=[(int(p[0]), int(p[1])) for p in c["points"]],
                        bbox=tuple(c["bbox"]),
                        area=float(c["area"]),
                        length=int(c["length"])
                    ))
            return result
        return self._find_python(gray)

    def _find_python(self, gray: LiteImage) -> List[ContourInfo]:
        binary = gray._data > 128
        if not binary.any():
            return []
        rows = np.where(binary.any(axis=1))[0]
        cols = np.where(binary.any(axis=0))[0]
        if len(rows) == 0:
            return []
        x, y = int(cols[0]), int(rows[0])
        w = int(cols[-1] - cols[0] + 1)
        h = int(rows[-1] - rows[0] + 1)
        area = float(binary.sum())
        if area < self.min_area:
            return []
        return [ContourInfo(
            points=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
            bbox=(x, y, w, h),
            area=area,
            length=2*(w+h)
        )]

    def filter_by_aspect(self, contours: List[ContourInfo],
                         min_ratio: float = 0.5, max_ratio: float = 2.0) -> List[ContourInfo]:
        """Filter contours by bounding box aspect ratio (w/h)."""
        result = []
        for c in contours:
            _, _, w, h = c.bbox
            ratio = w / h if h > 0 else 0
            if min_ratio <= ratio <= max_ratio:
                result.append(c)
        return result

    def filter_by_area(self, contours: List[ContourInfo],
                       min_area: float = 0, max_area: float = float("inf")) -> List[ContourInfo]:
        return [c for c in contours if min_area <= c.area <= max_area]


# ── HOG Object Detector ───────────────────────────────────────────────────────

class HOGMatcher:
    """
    Simple HOG-based object matcher using a reference image as the "model".

    Not as powerful as a trained SVM detector, but works without any
    training data or ML libraries — perfect for edge devices.

    Usage:
        matcher = HOGMatcher("reference.jpg", win_size=(64,128))
        detections = matcher.scan(scene, scale_factor=1.2, stride=8)
    """

    def __init__(self, reference: "str | LiteImage",
                 win_w: int = 64, win_h: int = 128,
                 threshold: float = 0.85):
        if isinstance(reference, str):
            from ._core import LiteImage as LI
            reference = LI.open(reference)
        self.win_w = win_w
        self.win_h = win_h
        self.threshold = threshold
        # Compute HOG descriptor of reference
        ref_resized = reference.to_gray().resize(win_w, win_h)
        if _HAS_EXT:
            self._ref_hog = np.array(_ext.hog_descriptor(ref_resized._data))
        else:
            self._ref_hog = self._compute_hog_numpy(ref_resized._data)
        # Normalize
        norm = np.linalg.norm(self._ref_hog)
        if norm > 0:
            self._ref_hog = self._ref_hog / norm

    def scan(self, scene: LiteImage, stride: int = 8,
             scales: List[float] = None) -> List[Detection]:
        """Multi-scale sliding window scan."""
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        detections = []
        for scale in scales:
            scaled = scene.scale(scale).to_gray()
            dets = self._scan_single(scaled, scale, stride)
            detections.extend(dets)
        return self._nms(detections)

    def _scan_single(self, scene: LiteImage, scale: float, stride: int) -> List[Detection]:
        detections = []
        h, w = scene.height, scene.width
        for y in range(0, h - self.win_h + 1, stride):
            for x in range(0, w - self.win_w + 1, stride):
                window = scene.crop(x, y, self.win_w, self.win_h)
                if _HAS_EXT:
                    desc = np.array(_ext.hog_descriptor(window._data))
                else:
                    desc = self._compute_hog_numpy(window._data)
                norm = np.linalg.norm(desc)
                if norm < 1e-6:
                    continue
                desc = desc / norm
                score = float(np.dot(self._ref_hog, desc[:len(self._ref_hog)]))
                if score >= self.threshold:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(self.win_w / scale)
                    orig_h = int(self.win_h / scale)
                    detections.append(Detection(orig_x, orig_y, orig_w, orig_h,
                                                score=score, label="match"))
        return detections

    def _nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Non-Maximum Suppression."""
        if not detections:
            return []
        detections.sort(key=lambda d: -d.score)
        kept = []
        for d in detections:
            dominated = any(d.iou(k) > iou_threshold for k in kept)
            if not dominated:
                kept.append(d)
        return kept

    def _compute_hog_numpy(self, gray: np.ndarray) -> np.ndarray:
        """Numpy HOG fallback."""
        gx = np.gradient(gray.astype(float), axis=1)
        gy = np.gradient(gray.astype(float), axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        ang = (np.degrees(np.arctan2(gy, gx)) % 180)
        cell = 8; bins = 9; h, w = gray.shape
        cells_x, cells_y = w // cell, h // cell
        hog = []
        for cy in range(cells_y):
            for cx in range(cells_x):
                hist = np.zeros(bins)
                patch_m = mag[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
                patch_a = ang[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
                for b in range(bins):
                    mask = (patch_a >= b*20) & (patch_a < (b+1)*20)
                    hist[b] = patch_m[mask].sum()
                hog.append(hist)
        return np.concatenate(hog)
