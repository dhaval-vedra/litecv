# LiteCV Usage Guide

## Install locally

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

## Import the library

```python
import litecv
from litecv import new_image, open_image, FilterType, StreamingFilter, RealTimeCameraApp
```

## Create and save an image

```python
img = new_image(320, 240, color='lightblue')
img.draw_text('LiteCV Demo', (20, 20), color='black', size=24)
img.draw_circle((160, 120), 50, color='red', fill='yellow')
img.save('demo_image.jpg')
```

## Apply filters

```python
img = open_image('demo_image.jpg')
gray = StreamingFilter(FilterType.GRAYSCALE).apply(img.copy())
gray.save('demo_image_gray.jpg')
```

## Camera capture

```python
app = RealTimeCameraApp(resolution=(640, 480), camera_resolution=(320, 240))
app.start()
```

Use keys 1-9 to switch filters and `ESC` to close the camera window.

## Example scripts

The `examples/` folder includes several demos:

- `examples/demo.py` — create an image, apply a filter, and launch the camera app.
- `examples/basic_image.py` — draw shapes and text to create a basic image.
- `examples/filters_demo.py` — exercise multiple built-in filters.
- `examples/camera_app.py` — start the realtime camera application.
- `examples/video_demo.py` — placeholder video frame output demo.
- `examples/object_detection.py` — object detection overlay example.
- `examples/utilities.py` — image concatenation and blending examples.

Run any example with:

```bash
python examples/<name>.py
```
