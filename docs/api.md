# LiteCV API Reference

## Package API

### `litecv.AdvancedLiteImage`
A core image wrapper with support for common image operations.

- `AdvancedLiteImage(path=...)`
- `resize(width, height, optimize_speed=True)`
- `resize_percent(percent, optimize_speed=True)`
- `to_gray(method='lightness')`
- `to_rgb()`
- `blur(radius=2, optimize_speed=True)`
- `sharpen(factor=2.0)`
- `brightness(factor)`
- `contrast(factor)`
- `saturation(factor)`
- `draw_text(text, position, color, size, font_path=None)`
- `draw_rectangle(top_left, bottom_right, color, width, fill=None)`
- `draw_circle(center, radius, color, width=2, fill=None)`
- `draw_line(start, end, color='red', width=2)`
- `rotate(degrees, expand=True, optimize_speed=True)`
- `save(path, format=None, optimize=True, quality=85)`

### `litecv.FilterType`
Enum values for available filters:
- `FilterType.GRAYSCALE`
- `FilterType.EDGES`
- `FilterType.BLUR`
- `FilterType.MOTION_DETECT`
- `FilterType.SEPIA`
- `FilterType.CARTOON`
- `FilterType.SKETCH`
- `FilterType.THERMAL`
- `FilterType.NIGHT_VISION`
- `FilterType.INFRARED`

### `litecv.StreamingFilter`
Apply realtime filters to an `AdvancedLiteImage`.

```python
filter = StreamingFilter(FilterType.GRAYSCALE)
output = filter.apply(image)
```

### `litecv.CameraFeed`
Capture frames from a camera using `pygame`.

```python
feed = CameraFeed(camera_id=0, resolution=(640, 480), fps=30)
feed.start()
frame = feed.get_frame()
feed.stop()
```

### `litecv.RealTimeCameraApp`
A basic realtime camera application with built-in filter controls.

```python
app = RealTimeCameraApp(resolution=(800, 600), camera_resolution=(640, 480))
app.start()
```

### Utility functions

- `open_image(path)` — open a file into `AdvancedLiteImage`
- `new_image(width, height, color='white')` — create a blank image
- `concatenate(images, direction='horizontal')` — join multiple images
- `blend_images(image1, image2, alpha=0.5)` — blend two images

### Logo utilities

- `get_logo_path(format='svg')` — get path to logo file ('svg', 'png', or 'ascii')
- `get_ascii_logo()` — get ASCII art logo as string

### Demo helpers

- `demo_basic_operations()`
- `demo_real_time_camera()`
- `demo_filters()`
