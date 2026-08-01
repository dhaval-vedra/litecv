"""
litecv._camera — Camera feed management for edge devices

Supported backends (in order of preference):
  1. C++ V4L2 mmap (zero-copy, Linux) — fastest
  2. picamera2 (Raspberry Pi Camera Module)
  3. Pure-Python V4L2 (slower, Python ctypes)

Features:
  - CameraFeed: synchronous iterator
  - AsyncCamera: threaded frame buffer (non-blocking for pipelines)
  - FrameBuffer: ring buffer for temporal operations
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Callable, Iterator

import numpy as np

try:
    import _lcv_ext as _ext
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

from ._core import LiteImage


class CameraFeed:
    """
    Synchronous camera feed — use as iterator or context manager.

    Example:
        with CameraFeed("/dev/video0", 640, 480) as cam:
            for frame in cam:
                frame.sepia().save("out.jpg")
                break
    """

    def __init__(self, device: str = "/dev/video0",
                 width: int = 640, height: int = 480,
                 prefer_mjpeg: bool = True,
                 backend: str = "auto"):
        self.device = device
        self.width = width
        self.height = height
        self.prefer_mjpeg = prefer_mjpeg
        self.backend = backend
        self._cam = None
        self._picam = None
        self._frame_count = 0
        self._start_time = None

    def open(self) -> "CameraFeed":
        backend = self.backend

        if backend in ("auto", "v4l2_cpp") and _HAS_EXT:
            try:
                self._cam = _ext.Camera(self.device, self.width, self.height,
                                        self.prefer_mjpeg)
                self.backend = "v4l2_cpp"
                self.width  = self._cam.width
                self.height = self._cam.height
                print(f"[litecv] Camera: V4L2 C++ ({self._cam.format}) "
                      f"{self.width}×{self.height}")
                self._start_time = time.time()
                return self
            except Exception as e:
                if backend == "v4l2_cpp":
                    raise
                print(f"[litecv] V4L2 C++ failed ({e}), trying picamera2…")

        if backend in ("auto", "picamera2"):
            try:
                from picamera2 import Picamera2  # type: ignore
                self._picam = Picamera2()
                config = self._picam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"})
                self._picam.configure(config)
                self._picam.start()
                self.backend = "picamera2"
                print(f"[litecv] Camera: picamera2 {self.width}×{self.height}")
                self._start_time = time.time()
                return self
            except Exception as e:
                if backend == "picamera2":
                    raise
                print(f"[litecv] picamera2 failed ({e})")

        raise RuntimeError(
            f"No camera backend available for {self.device}. "
            "Install pybind11 + compile C++ extension, or use picamera2.")

    def close(self):
        self._cam   = None
        if self._picam:
            try:
                self._picam.stop()
            except Exception:
                pass
            self._picam = None

    def __enter__(self) -> "CameraFeed":
        return self.open()

    def __exit__(self, *args):
        self.close()

    def capture_raw(self) -> np.ndarray:
        """Capture one frame as RGB numpy array."""
        if self._cam:
            return self._cam.capture()
        if self._picam:
            return self._picam.capture_array()
        raise RuntimeError("Camera not opened — call open() first")

    def capture(self) -> LiteImage:
        """Capture one frame as LiteImage."""
        self._frame_count += 1
        return LiteImage(self.capture_raw())

    def fps(self) -> float:
        if not self._start_time or self._frame_count == 0:
            return 0.0
        return self._frame_count / (time.time() - self._start_time)

    def __iter__(self) -> Iterator[LiteImage]:
        return self

    def __next__(self) -> LiteImage:
        return self.capture()


class AsyncCamera:
    """
    Non-blocking camera with a background capture thread.

    Maintains a ring buffer of the last `buffer_size` frames.
    Calling get_frame() returns the most recent frame immediately
    without waiting for the next capture.

    Perfect for processing pipelines where the camera should always
    run at max FPS regardless of processing speed.

    Example:
        cam = AsyncCamera("/dev/video0", 640, 480, buffer_size=3)
        cam.start()
        for _ in range(100):
            frame = cam.get_frame()   # always fresh, non-blocking
            processed = frame.sepia()
        cam.stop()
    """

    def __init__(self, device: str = "/dev/video0",
                 width: int = 640, height: int = 480,
                 buffer_size: int = 3,
                 prefer_mjpeg: bool = True):
        self._feed = CameraFeed(device, width, height, prefer_mjpeg)
        self._buffer: deque[LiteImage] = deque(maxlen=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._error: Optional[Exception] = None

    def start(self) -> "AsyncCamera":
        self._feed.open()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name="litecv-cam")
        self._thread.start()
        # Wait for first frame
        self._frame_event.wait(timeout=5.0)
        if self._error:
            raise self._error
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._feed.close()

    def get_frame(self) -> Optional[LiteImage]:
        """Get the most recent captured frame (non-blocking)."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def get_frames(self, n: int) -> list:
        """Get the last n frames (for temporal operations)."""
        with self._lock:
            return list(self._buffer)[-n:]

    @property
    def fps(self) -> float:
        return self._feed.fps()

    @property
    def width(self) -> int:
        return self._feed.width

    @property
    def height(self) -> int:
        return self._feed.height

    def __enter__(self) -> "AsyncCamera":
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def _capture_loop(self):
        while self._running:
            try:
                frame = self._feed.capture()
                with self._lock:
                    self._buffer.append(frame)
                self._frame_event.set()
            except Exception as e:
                self._error = e
                self._frame_event.set()
                break


class FrameBuffer:
    """
    Ring buffer for temporal image operations.

    Use for:
    - Background subtraction (compare with N frames ago)
    - Frame averaging (noise reduction)
    - Motion detection (frame difference)
    - Temporal smoothing

    Example:
        buf = FrameBuffer(capacity=10)
        for frame in cam:
            buf.push(frame)
            if buf.full:
                diff = frame.absdiff(buf.get(-5))  # diff with 5 frames ago
    """

    def __init__(self, capacity: int = 30):
        self._buf: deque[LiteImage] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, frame: LiteImage):
        self._buf.append(frame)

    def get(self, index: int) -> Optional[LiteImage]:
        """Get frame at index (0=oldest, -1=newest)."""
        if not self._buf:
            return None
        try:
            return self._buf[index]
        except IndexError:
            return None

    @property
    def full(self) -> bool:
        return len(self._buf) == self.capacity

    @property
    def size(self) -> int:
        return len(self._buf)

    def average(self) -> Optional[LiteImage]:
        """Temporal average of all buffered frames (noise reduction)."""
        if not self._buf:
            return None
        acc = self._buf[0]._data.astype(float)
        for f in list(self._buf)[1:]:
            acc += f._data.astype(float)
        return LiteImage((acc / len(self._buf)).astype(np.uint8))

    def motion_mask(self, threshold: int = 25) -> Optional[LiteImage]:
        """Difference between newest and oldest frame as binary mask."""
        if len(self._buf) < 2:
            return None
        diff = self._buf[-1].absdiff(self._buf[0]).to_gray()
        return diff.threshold(threshold)
