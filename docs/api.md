# litecv-advanced — Full API Reference

## Module: `litecv`

### Top-level functions

| Function | Description |
|---|---|
| `open_image(path, grayscale=False)` | Load image from disk → LiteImage |
| `from_numpy(array)` | Create from uint8 numpy array |
| `from_bytes(data)` | Decode JPEG/PNG bytes → LiteImage |
| `new_image(w, h, color, channels)` | Create blank image |
| `backend_info()` | Dict: version, SIMD, C++ ext, Pillow, etc. |

---

## Class: `LiteImage`

### Properties
| Property | Type | Description |
|---|---|---|
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `channels` | int | 1 (gray) or 3 (RGB) |
| `shape` | tuple | numpy shape (H, W) or (H, W, 3) |
| `is_gray` | bool | True if grayscale |
| `data` | np.ndarray | Raw pixel data |

### I/O
| Method | Returns | Description |
|---|---|---|
| `.open(path, grayscale)` | LiteImage | [static] Load from file |
| `.save(path, quality=90)` | self | Save to JPEG/PNG |
| `.to_bytes(fmt=".jpg")` | bytes | Encode to bytes |
| `.from_bytes(data)` | LiteImage | [static] Decode from bytes |
| `.copy()` | LiteImage | Deep copy |

### Geometry
| Method | Returns | Description |
|---|---|---|
| `.resize(w, h)` | LiteImage | Resize to exact dimensions |
| `.scale(fx, fy)` | LiteImage | Scale by factor |
| `.crop(x, y, w, h)` | LiteImage | Crop region |
| `.roi(x, y, w, h)` | LiteImage | Alias for crop |
| `.pad(top, bottom, left, right, value)` | LiteImage | Add border |
| `.rotate(degrees, expand)` | LiteImage | Rotate (any angle) |
| `.rotate90(times)` | LiteImage | Rotate 90°×N (fast SIMD) |
| `.flip(direction)` | LiteImage | FlipDir.HORIZONTAL/VERTICAL/BOTH |
| `.transpose()` | LiteImage | Swap axes |

### Color
| Method | Returns | Description |
|---|---|---|
| `.to_gray()` | LiteImage | RGB → grayscale (SIMD) |
| `.to_rgb()` | LiteImage | Gray → RGB (3ch) |
| `.split_channels()` | (R, G, B) | Split into 3 LiteImages |
| `LiteImage.merge_channels(r, g, b)` | LiteImage | Merge 3 grayscale images |

### Adjustments
| Method | Returns | Description |
|---|---|---|
| `.brightness(delta)` | LiteImage | Add delta (SIMD, clamped) |
| `.contrast(scale)` | LiteImage | Scale around 128 |
| `.gamma(value)` | LiteImage | Gamma correction |
| `.invert()` | LiteImage | 255 - pixel |
| `.normalize()` | LiteImage | Stretch to 0-255 |
| `.threshold(thresh, maxval, invert)` | LiteImage | Binary threshold |
| `.threshold_otsu()` | LiteImage | Automatic Otsu threshold |
| `.absdiff(other)` | LiteImage | \|self - other\| (SIMD) |

### Filters
| Method | Returns | Description |
|---|---|---|
| `.blur(radius)` | LiteImage | Gaussian blur |
| `.gaussian_blur(sigma)` | LiteImage | Gaussian blur by sigma |
| `.median(ksize)` | LiteImage | Median filter |
| `.bilateral(d, sigma_color, sigma_space)` | LiteImage | Edge-preserving blur (C++) |
| `.sharpen()` | LiteImage | Unsharp sharpen |
| `.unsharp_mask(amount, radius)` | LiteImage | Unsharp mask (C++) |
| `.edges()` | LiteImage | Sobel edge detection |
| `.laplacian()` | LiteImage | Laplacian edge |
| `.emboss()` | LiteImage | Emboss effect |
| `.denoise()` | LiteImage | Smooth + denoise |
| `.clahe(tile_w, tile_h, clip_limit)` | LiteImage | CLAHE low-light enhance (C++) |
| `.equalize()` | LiteImage | Global histogram equalization |
| `.filter(FilterType, ksize)` | LiteImage | Generic filter dispatcher |
| `.custom_kernel(kernel, scale, offset)` | LiteImage | Custom convolution |

### Morphological (on grayscale/binary)
| Method | Returns | Description |
|---|---|---|
| `.erode(ksize)` | LiteImage | Erosion (min filter) |
| `.dilate(ksize)` | LiteImage | Dilation (max filter) |
| `.morph(op, ksize, shape)` | LiteImage | Generic morphological op |

**op values:** `"erode"`, `"dilate"`, `"open"`, `"close"`, `"gradient"`, `"tophat"`, `"blackhat"`
**shape values:** `"rect"` (default), `"ellipse"`, `"cross"`

### Effects (C++ or Pillow fallback)
| Method | Returns | Description |
|---|---|---|
| `.sepia()` | LiteImage | Sepia tone |
| `.thermal()` | LiteImage | Iron colormap |
| `.night_vision()` | LiteImage | CLAHE + green channel |
| `.sketch()` | LiteImage | Pencil sketch |
| `.cartoon()` | LiteImage | Cartoon effect |
| `.edge_glow(color)` | LiteImage | Colored edge highlight |

### Drawing
| Method | Returns | Description |
|---|---|---|
| `.draw_rect(x, y, w, h, color, thickness)` | LiteImage | Draw rectangle |
| `.draw_circle(cx, cy, r, color, thickness)` | LiteImage | Draw circle |
| `.draw_line(x1, y1, x2, y2, color, thickness)` | LiteImage | Draw line |
| `.draw_text(text, x, y, color)` | LiteImage | Draw text |
| `.draw_keypoints(kps, color, radius)` | LiteImage | Draw feature points |
| `.draw_contours(contours, color)` | LiteImage | Draw contour points |

### Analysis
| Method | Returns | Description |
|---|---|---|
| `.histogram()` | list[int] | 256-bin histogram |
| `.dominant_colors(n)` | list[dict] | Top N colors with hex+fraction |
| `.mean_color()` | tuple | Per-channel mean |
| `.std_color()` | tuple | Per-channel std dev |

### Pipeline operator
```python
result = img | P.clahe() | P.sepia()  # same as img.clahe().sepia()
```

---

## Module: `litecv.pipeline`

### Step functions (all return callable)
`blur(r)`, `gaussian_blur(s)`, `median(k)`, `bilateral(d,sc,ss)`, `sharpen()`,
`unsharp_mask(amt,r)`, `edges()`, `laplacian()`, `emboss()`, `denoise()`,
`clahe(tw,th,cl)`, `equalize()`, `resize(w,h)`, `scale(fx,fy)`, `crop(x,y,w,h)`,
`roi(x,y,w,h)`, `pad(t,b,l,r,v)`, `rotate(deg)`, `rotate90(n)`, `flip(dir)`,
`transpose()`, `to_gray()`, `to_rgb()`, `brightness(d)`, `contrast(s)`,
`gamma(v)`, `invert()`, `normalize()`, `threshold(t)`, `threshold_otsu()`,
`sepia()`, `thermal()`, `night_vision()`, `sketch()`, `cartoon()`, `edge_glow(c)`,
`morph(op,k,sh)`, `erode(k)`, `dilate(k)`,
`save(path)`, `save_dir(dir,prefix,ext)`, `tap(fn)`, `branch(*pipelines)`, `copy()`

### Class: `Pipeline`
| Method | Description |
|---|---|
| `Pipeline([steps])` | Create pipeline from step list |
| `p(img)` | Run on one image |
| `p.map(images)` | Apply to list of images |
| `p1 >> p2` | Compose pipelines |
| `p.benchmark(img, n=10)` | Timing stats dict |
| `img \| step` | Pipe operator |

### Preset pipelines
| Function | Description |
|---|---|
| `enhance_camera()` | CLAHE + denoise + unsharp |
| `surveillance()` | denoise + sharpen + equalize |
| `night_mode()` | CLAHE + night_vision + unsharp |
| `document_scan()` | gray + CLAHE + Otsu threshold |

---

## Class: `CameraFeed`
```python
CameraFeed(device="/dev/video0", width=640, height=480,
           prefer_mjpeg=True, backend="auto")
```
| Method | Description |
|---|---|
| `.open()` | Open camera (returns self) |
| `.close()` | Close camera |
| `.capture()` → LiteImage | Capture one frame |
| `.capture_raw()` → np.ndarray | Raw RGB array |
| `.fps` → float | Current FPS |
| context manager | `with CameraFeed(...) as cam:` |
| iteration | `for frame in cam:` |

**Backends:** `v4l2_cpp` (zero-copy mmap) → `picamera2` (Pi Camera)

---

## Class: `AsyncCamera`
```python
AsyncCamera(device, width, height, buffer_size=3, prefer_mjpeg=True)
```
| Method | Description |
|---|---|
| `.start()` | Start background capture thread |
| `.stop()` | Stop camera |
| `.get_frame()` → LiteImage | Most recent frame (non-blocking) |
| `.get_frames(n)` → list | Last n frames |
| `.fps` → float | Camera capture FPS |

---

## Class: `FrameBuffer`
```python
FrameBuffer(capacity=30)
```
| Method | Description |
|---|---|
| `.push(frame)` | Add frame to buffer |
| `.get(index)` | Get frame (0=oldest, -1=newest) |
| `.average()` | Temporal average of all frames |
| `.motion_mask(threshold)` | Difference newest-oldest → binary mask |
| `.full` | True when capacity reached |

---

## Class: `HarrisDetector`
```python
HarrisDetector(max_corners=500, threshold=0.01, block_size=3, k=0.04)
detector.detect(img) → List[KeyPoint]
```
**KeyPoint:** `.x`, `.y`, `.score`, `.as_tuple()`

---

## Class: `OpticalFlow`
```python
OpticalFlow(win_size=15, max_iter=30, eps=0.01)
flow.calc(prev, next, points) → List[FlowVector]
flow.draw_flow(img, vectors, color, scale) → LiteImage
```
**FlowVector:** `.x`, `.y`, `.u`, `.v`, `.valid`, `.magnitude`, `.direction_deg`, `.end_point`

---

## Class: `MotionDetector`
```python
MotionDetector(threshold=25, min_area=500, learning_rate=0.05, morph_ksize=5)
detector.detect(frame) → List[Detection]
detector.reset()
```
**Detection:** `.x`, `.y`, `.w`, `.h`, `.score`, `.label`, `.cx`, `.cy`, `.area`, `.iou(other)`

---

## Class: `TemplateMatcher`
```python
TemplateMatcher(template, method="ncc", threshold=0.7)
matcher.match(scene) → Optional[Detection]
```
**method:** `"ncc"` (Normalized Cross Correlation) or `"ssd"` (Sum of Squared Differences)

---

## Class: `ContourAnalyzer`
```python
ContourAnalyzer(min_area=100, min_length=10)
analyzer.find(mask) → List[ContourInfo]
analyzer.filter_by_aspect(contours, min_ratio, max_ratio)
analyzer.filter_by_area(contours, min_area, max_area)
```
**ContourInfo:** `.points`, `.bbox` (x,y,w,h), `.area`, `.length`

---

## Class: `HOGMatcher`
```python
HOGMatcher(reference, win_w=64, win_h=128, threshold=0.85)
matcher.scan(scene, stride=8, scales=[0.5,0.75,1.0,1.25,1.5]) → List[Detection]
```

---

## Class: `MJPEGServer`
```python
MJPEGServer(host="0.0.0.0", port=8080, quality=70, max_fps=30)
```
| Method | Description |
|---|---|
| `.start()` | Start HTTP server (non-blocking) |
| `.stop()` | Stop server |
| `.push_frame(img)` | Push frame to all clients |
| `.start_feed(camera, process_fn, max_fps)` | Auto-capture + push in background |
| `.url` | Stream URL string |
| `.stats` | Dict: fps, frames, uptime |

**Endpoints:** `/stream` (MJPEG), `/frame.jpg` (snapshot), `/` (status page)

---

## Class: `Profiler`
```python
prof = Profiler(window=100)
with prof.step("blur"): img.blur()
prof.frame()            # call once per frame
prof.stats() → dict     # per-step statistics
prof.report()           # print formatted table
prof.reset()
```

## Function: `fps_counter(window=30)`
Returns a callable. Call it every frame to get current FPS.

## Class: `MemoryMonitor`
```python
mem = MemoryMonitor()
mem.rss_mb    # RSS memory in MB
mem.peak_mb   # Peak virtual memory
mem.report()  # Print + return string
```

---

## ML Bridge

### `TFLiteModel(model_path, num_threads=4)`
| Method | Description |
|---|---|
| `.run(img)` | Raw inference → list of output arrays |
| `.classify(img, top_k, labels)` | → [(label, score)] |
| `.detect(img, conf_threshold)` | → [{x,y,w,h,score,class_id}] |
| `.preprocess(img)` | → normalized numpy array |

### `ONNXModel(model_path, input_shape, providers)`
| Method | Description |
|---|---|
| `.run(img)` | Raw inference → list of arrays |
| `.preprocess(img)` | → CHW float32 array |

### `auto_model(model_path)` → best available model

---

## C++ Extension: `_lcv_ext`

Direct access (rarely needed — use LiteImage instead):

```python
import _lcv_ext as ext
ext.simd_backend()          # "ARM NEON" / "x86 AVX2" / "scalar"
ext.pool_stats()            # memory pool usage
ext.clahe(arr, tw, th, cl)
ext.bilateral(arr, d, sc, ss)
ext.morph(arr, op, ksize, shape)
ext.harris_corners(gray, bs, ksize, k, thr, max_n) → [(x,y,score)]
ext.lk_optical_flow(prev, next, pts, ws, mi, eps) → [(x,y,u,v,valid)]
ext.hog_descriptor(gray, cell_size, block_size, num_bins) → [float]
ext.template_match(img, tmpl, method) → (x, y, score)
ext.find_contours(mask, min_len) → [{points, bbox, area, length}]
ext.dominant_colors(rgb, n) → [{r,g,b,fraction,hex}]
ext.brightness(arr, delta)
ext.contrast(arr, scale)
ext.absdiff(a, b)
ext.to_gray(rgb_arr)
ext.imread(path, grayscale) / ext.imwrite(path, arr)
ext.imdecode(bytes) / ext.imencode(arr, ext)
ext.resize(arr, w, h) / ext.rotate(arr, code) / ext.flip(arr, code)
ext.effect_sepia(arr) / ext.effect_thermal(arr) / ...
ext.Camera(device, w, h, prefer_mjpeg)
ext.BackgroundSubtractor(lr, thr)
```
