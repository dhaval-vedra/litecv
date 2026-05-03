# LiteCV Examples

This folder contains example scripts demonstrating various features of the LiteCV library.

## Available Examples

### Basic Usage
- `demo.py` — Complete demo with image creation, filters, and camera app
- `basic_image.py` — Draw shapes, text, and save images
- `logo_demo.py` — Display and use LiteCV logos

### Filters and Effects
- `filters_demo.py` — Apply multiple built-in filters and save results

### Advanced Features
- `camera_app.py` — Launch the realtime camera application
- `video_demo.py` — Placeholder video frame processing demo
- `object_detection.py` — Object detection and overlay example
- `utilities.py` — Image concatenation and blending examples

## Running Examples

All examples can be run directly with Python:

```bash
python examples/basic_image.py
python examples/filters_demo.py
python examples/logo_demo.py
```

## Notes

- Camera examples (`camera_app.py`) require a webcam and will open a Pygame window
- Some examples generate output files in the current directory
- The `object_detection.py` example creates a sample image if none exists

## Logo Usage

The library includes logo assets that can be accessed programmatically:

```python
from litecv import get_logo_path, get_ascii_logo

# Get logo file paths
svg_path = get_logo_path('svg')
png_path = get_logo_path('png')

# Get ASCII logo as string
ascii_logo = get_ascii_logo()
print(ascii_logo)
```