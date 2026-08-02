# litecv

**Lightweight yet powerful computer vision library for edge devices**
Raspberry Pi · Jetson Nano · Any ARM/x86 Linux board

> No OpenCV. No pygame. No heavy ML frameworks.
> Just Pillow + NumPy + optional C++ extension with SIMD acceleration.

---


<!-- QUICK INSTALL -->
<div align="center">

### ⚡ Get Started in Seconds

```bash
pip install litecv
```

*No OpenCV. No pygame. No heavy ML frameworks.*

</div>

## Why litecv-advanced?

| Feature | OpenCV | litecv-advanced |
|---|---|---|
| Install size | ~80 MB | ~2 MB (Python) / ~5 MB (with C++) |
| Dependencies | 50+ libs | Pillow + NumPy only |
| ARM NEON SIMD | ✓ | ✓ (auto-detect) |
| Zero-copy V4L2 camera | Partial | ✓ (mmap DMA) |
| CLAHE | ✓ | ✓ (C++) |
| Bilateral filter | ✓ | ✓ (C++) |
| Harris corners | ✓ | ✓ (C++) |
| Lucas-Kanade flow | ✓ | ✓ (C++) |
| Morphological ops | ✓ | ✓ (7 operations) |
| HOG descriptor | ✓ | ✓ (C++) |
| Background subtraction | ✓ | ✓ (C++) |
| Contour detection | ✓ | ✓ (C++) |
| Template matching | ✓ | ✓ (SSD + NCC) |
| MJPEG HTTP streaming | ✗ | ✓ (built-in) |
| Pipeline API | ✗ | ✓ |
| TFLite bridge | ✗ | ✓ (optional) |
| ONNX bridge | ✗ | ✓ (optional) |
| Async camera | ✗ | ✓ |
| Memory pool (no malloc) | Partial | ✓ |
| Pure-Python fallback | ✗ | ✓ |
| pygame | ✗ | ✗ (intentionally removed) |

---

## Installation

### Minimum (Pillow + NumPy only — pure Python)
```bash
pip install pillow numpy
pip install . --no-build-isolation
```

### Recommended (with C++ extension — 5-10x faster)
```bash
# Raspberry Pi / Debian
sudo apt-get install cmake g++ python3-dev

# Any platform
pip install pybind11
pip install .
```

### Optional ML support
```bash
pip install .[ml]      # TFLite (Raspberry Pi optimised)
pip install .[onnx]    # ONNX Runtime
pip install .[camera]  # picamera2 (Pi Camera Module)
```

---

## Quick Start

```python
import litecv

# Load and process
img = litecv.open_image("photo.jpg")
img.clahe().sepia().save("result.jpg")

# Check what's running
print(litecv.backend_info())
# {'version': '2.0.0', 'cpp_extension': True, 'simd': 'ARM NEON', ...}
```

---

## Features

### 1. Image Operations

```python
import litecv
from litecv import open_image, new_image, from_numpy

img = open_image("photo.jpg")

# Geometry
img.resize(640, 480)
img.scale(0.5)
img.crop(x=100, y=50, w=200, h=150)
img.roi(100, 50, 200, 150)          # alias for crop
img.pad(top=10, bottom=10, left=5, right=5)
img.rotate(90)
img.rotate90(times=1)
img.flip()                           # horizontal
img.flip("vertical")
img.transpose()

# Color
img.to_gray()
img.to_rgb()
r, g, b = img.split_channels()
merged  = litecv.LiteImage.merge_channels(r, g, b)

# Adjustments (SIMD-accelerated)
img.brightness(50)         # +50 to all pixels
img.contrast(1.5)          # scale around mid-point
img.gamma(0.8)
img.invert()
img.normalize()
img.threshold(128)
img.threshold_otsu()       # automatic threshold
img.absdiff(other)         # |img - other|

# Filters
img.blur(radius=2)
img.gaussian_blur(sigma=1.5)
img.median(ksize=5)
img.bilateral(d=9, sigma_color=75, sigma_space=75)   # edge-preserving
img.sharpen()
img.unsharp_mask(amount=1.5, radius=3)
img.edges()                # Sobel
img.laplacian()
img.emboss()
img.denoise()

# Advanced filters
img.clahe(tile_w=8, tile_h=8, clip_limit=2.0)  # low-light enhancement
img.equalize()             # global histogram equalization

# Morphological (on grayscale/binary)
mask = img.to_gray().threshold(128)
mask.erode(ksize=3)
mask.dilate(ksize=3)
mask.morph("open",     ksize=5)
mask.morph("close",    ksize=5)
mask.morph("gradient", ksize=3)
mask.morph("tophat",   ksize=5)
mask.morph("blackhat", ksize=5)
mask.morph("erode", ksize=5, shape="ellipse")
mask.morph("dilate", ksize=5, shape="cross")

# Effects
img.sepia()
img.thermal()              # iron colormap
img.night_vision()         # CLAHE + green channel
img.sketch()               # edge inversion
img.cartoon()              # bilateral + quantize + edge mask
img.edge_glow(color=(0, 255, 120))

# Drawing
img.draw_rect(x, y, w, h, color=(255,0,0), thickness=2)
img.draw_circle(cx, cy, radius, color=(0,255,0), thickness=2)
img.draw_line(x1, y1, x2, y2, color=(255,255,0), thickness=1)
img.draw_text("Hello", x, y, color=(255,255,255))
img.draw_keypoints([(x,y,score),...], color=(0,255,0), radius=4)
img.draw_contours(contours, color=(0,255,0))

# Analysis
img.histogram()             # 256-bin list
img.dominant_colors(n=5)    # [{'r':..,'g':..,'b':..,'hex':..,'fraction':..}]
img.mean_color()
img.std_color()
```

### 2. Pipeline API

```python
from litecv import pipeline as P, Pipeline

# Operator style
result = img | P.clahe() | P.sepia() | P.save("out.jpg")

# Pipeline object (reusable)
enhance = Pipeline([P.clahe(clip_limit=3.0), P.unsharp_mask(), P.contrast(1.3)])
result = enhance(img)

# Apply to many images
results = enhance.map(list_of_images)

# Compose pipelines
full = enhance >> Pipeline([P.sepia()])

# Benchmark
stats = enhance.benchmark(img, n=100)
print(f"FPS: {stats['fps']:.1f}")

# Preset pipelines
from litecv.pipeline import enhance_camera, night_mode, document_scan
result = enhance_camera()(img)   # denoise + clahe + unsharp
result = night_mode()(img)       # clahe + night_vision
result = document_scan()(img)    # grayscale + clahe + otsu threshold
```

### 3. Camera (Real-Time, Zero-Copy)

```python
from litecv import CameraFeed, AsyncCamera, FrameBuffer

# Synchronous (simple)
with CameraFeed("/dev/video0", 640, 480) as cam:
    for frame in cam:
        processed = frame.clahe().thermal()
        processed.save("frame.jpg")
        break

# Async (non-blocking — camera runs at max FPS independently)
with AsyncCamera("/dev/video0", 640, 480, buffer_size=3) as cam:
    while True:
        frame = cam.get_frame()  # always fresh, never blocks
        if frame:
            frame.sepia().save("async.jpg")
        print(f"Camera FPS: {cam.fps:.1f}")

# Temporal operations with FrameBuffer
buf = FrameBuffer(capacity=30)
with CameraFeed("/dev/video0") as cam:
    for frame in cam:
        buf.push(frame)
        if buf.full:
            motion = buf.motion_mask(threshold=20)  # binary motion mask
            avg    = buf.average()                   # temporal average (noise ↓)
```

### 4. Detection

```python
from litecv import HarrisDetector, OpticalFlow, MotionDetector, TemplateMatcher

# Harris corner detection
detector = HarrisDetector(max_corners=500, threshold=0.01)
corners  = detector.detect(img)
img = img.draw_keypoints([(c.x, c.y) for c in corners])

# Lucas-Kanade optical flow
pts  = [(c.x, c.y) for c in corners[:100]]
flow = OpticalFlow(win_size=15)
with CameraFeed("/dev/video0") as cam:
    prev = cam.capture()
    for curr in cam:
        vectors = flow.calc(prev, curr, pts)
        moving  = [v for v in vectors if v.magnitude > 2.0]
        curr = flow.draw_flow(curr, vectors, scale=3.0)
        prev = curr

# Motion detection (background subtraction)
motion = MotionDetector(threshold=25, min_area=500)
with CameraFeed("/dev/video0") as cam:
    for frame in cam:
        detections = motion.detect(frame)
        for d in detections:
            frame = frame.draw_rect(d.x, d.y, d.w, d.h, (255,0,0))

# Template matching
matcher = TemplateMatcher("template.jpg", method="ncc", threshold=0.8)
result  = matcher.match(scene)
if result:
    scene = scene.draw_rect(result.x, result.y,
                            matcher.template.width, matcher.template.height)
```

### 5. MJPEG Streaming

```python
from litecv import MJPEGServer, CameraFeed

cam = CameraFeed("/dev/video0", 640, 480)
cam.open()

with MJPEGServer(port=8080) as server:
    # Stream with processing applied
    server.start_feed(cam, process_fn=lambda f: f.night_vision(), max_fps=30)
    print(f"Stream: {server.url}")     # http://0.0.0.0:8080/stream
    input("Press Enter to stop…")

# Or manually push frames:
server = MJPEGServer(port=8080).start()
while True:
    frame = cam.capture()
    server.push_frame(frame.thermal())
```

Open in browser: `http://raspberrypi.local:8080/`

### 6. ML Inference (Optional)

```python
from litecv.ml import TFLiteModel, ONNXModel, auto_model

# TFLite (best for Raspberry Pi)
model = TFLiteModel("efficientnet_lite0.tflite", num_threads=4)
labels = model.classify(img, top_k=5)     # [(label, score)]
dets   = model.detect(img, conf=0.5)      # [{x,y,w,h,score,class_id}]

# ONNX Runtime
model = ONNXModel("yolov5s.onnx", input_shape=(1,3,640,640))
output = model.run(img)

# Auto-select
model = auto_model("model.tflite")        # picks best backend
```

### 7. Performance Profiling

```python
from litecv import Profiler, fps_counter, MemoryMonitor

# Per-step profiler
prof = Profiler(window=100)
with CameraFeed("/dev/video0") as cam:
    for frame in cam:
        with prof.step("capture"): frame = cam.capture()
        with prof.step("clahe"):   frame = frame.clahe()
        with prof.step("thermal"): result = frame.thermal()
        prof.frame()

prof.report()
# ─────────────────────────────────────────────────
# Step                Mean(ms)   Min(ms)   Max(ms)    FPS    Count
# ─────────────────────────────────────────────────
# thermal               2.13       1.98      4.51   469.5     100
# clahe                 1.87       1.72      3.20   534.4     100
# capture               8.33       7.91     12.40   120.0     100
# Overall FPS: 95.2 | Frames: 100

# Simple FPS
fps = fps_counter(window=30)
for frame in cam:
    print(f"FPS: {fps():.1f}", end="\r")

# Memory
mem = MemoryMonitor()
print(f"RSS: {mem.rss_mb:.1f} MB")
```

---

## C++ Extension — Architecture

```
src/
├── lcv_bindings.cpp   ← pybind11 module (main entry)
├── lcv_simd.hpp       ← ARM NEON + x86 SSE/AVX auto-detect
├── lcv_mempool.hpp    ← Thread-local bump allocator (no malloc per frame)
├── lcv_advanced.hpp   ← CLAHE, bilateral, Harris, LK flow, HOG, BGSub, etc.
├── lcv_morph.hpp      ← Erode/dilate/open/close/gradient/tophat/blackhat
├── lcv_camera.hpp     ← V4L2 mmap zero-copy camera (Linux)
└── liteCV/            ← liteCV6 headers (core, imgcodecs/stb, imgproc)
```

### Build
```bash
cd src && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 4
```

### Raspberry Pi NEON check
```python
import litecv
print(litecv.backend_info()["simd"])  # → "ARM NEON"
```

---

## Run Examples

```bash
# Basic image operations (no camera needed)
python examples/basic_image.py

# Pipeline API demo (no camera needed)
python examples/pipeline_demo.py

# Detection demo (no camera needed — uses synthetic images)
python examples/detection_demo.py

# Camera + streaming (needs /dev/video0)
python examples/camera_advanced.py --device /dev/video0 --effect thermal --stream

# Streaming demo with synthetic frames (no camera needed)
python examples/streaming_demo.py --no-camera --effect cartoon
# Open http://localhost:8080/ in browser
```

---

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE)
