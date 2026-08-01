# litecv-advanced Examples

## Run without a camera (testing on any machine)

```bash
cd litecv-advanced

# Basic image ops — creates several *.jpg files
python examples/basic_image.py

# Pipeline API demo — creates demo_*.jpg files
python examples/pipeline_demo.py

# Detection demo — creates detect_*.jpg files
python examples/detection_demo.py

# Streaming demo (no camera) — open http://localhost:8080/ in browser
python examples/streaming_demo.py --no-camera --effect cartoon
```

## Run with a USB/Pi camera

```bash
# List available cameras
ls /dev/video*

# Camera with profiling
python examples/camera_advanced.py --device /dev/video0 --effect thermal --frames 200

# Camera with HTTP streaming (open browser at http://hostname:8080/)
python examples/camera_advanced.py --device /dev/video0 --stream --effect night

# Async camera (non-blocking)
python examples/camera_advanced.py --device /dev/video0 --async_cam --effect cartoon

# Save frames to directory
python examples/camera_advanced.py --device /dev/video0 --save ./frames/

# Motion detection
python examples/camera_advanced.py --device /dev/video0 --motion

# Stream with custom effect
python examples/streaming_demo.py --device /dev/video0 --effect sepia --port 8080
```

## Available effects

| Effect | Description |
|---|---|
| `none` | No effect (raw) |
| `clahe` | CLAHE — contrast enhance (great for dark rooms) |
| `sepia` | Warm sepia tone |
| `thermal` | Thermal/heat camera simulation |
| `night` | Night vision (green, enhanced) |
| `sketch` | Pencil sketch |
| `cartoon` | Cartoon effect |

## Example files created

After running `basic_image.py`:
- `basic_canvas.jpg` — drawing demo
- `basic_effect_sepia.jpg`, `basic_effect_thermal.jpg`, etc.

After running `pipeline_demo.py`:
- `demo_source.jpg`, `demo_sepia.jpg`, `demo_thermal.jpg`, `demo_enhanced.jpg`, etc.

After running `detection_demo.py`:
- `detect_corners.jpg` — Harris corners visualized
- `detect_flow.jpg` — optical flow vectors
- `detect_motion_*.jpg` — motion detection frames
- `detect_match_result.jpg` — template matching result
- `detect_contours.jpg` — contour visualization
