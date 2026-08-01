"""
detection_demo.py — All detection features demo

Demonstrates:
  - Harris corner detection
  - Lucas-Kanade optical flow
  - Motion detection (background subtraction)
  - Template matching
  - Contour detection
  - HOG descriptor

Run:
    python examples/detection_demo.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import litecv
from litecv import (
    from_numpy, open_image,
    HarrisDetector, OpticalFlow, MotionDetector,
    TemplateMatcher, ContourAnalyzer
)

try:
    import _lcv_ext as ext
    print(f"C++ ext: {ext.simd_backend()}")
except ImportError:
    ext = None
    print("Pure Python mode")


# ── Create synthetic test images ──────────────────────────────────────────────
def make_checkerboard(w=320, h=240, cell=32):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if (x//cell + y//cell) % 2 == 0:
                arr[y, x] = [200, 200, 200]
            else:
                arr[y, x] = [50, 50, 100]
    return from_numpy(arr)

def make_motion_frame(w=320, h=240, ball_x=100, ball_y=120, r=30):
    arr = np.full((h, w, 3), 80, dtype=np.uint8)
    cy, cx = np.ogrid[:h, :w]
    mask = (cx-ball_x)**2 + (cy-ball_y)**2 <= r**2
    arr[mask] = [255, 100, 0]
    return from_numpy(arr)


# ── 1. Harris Corner Detection ────────────────────────────────────────────────
print("\n─── Harris Corner Detection ───")
board = make_checkerboard()
detector = HarrisDetector(max_corners=200, threshold=0.02)

t0 = time.perf_counter()
corners = detector.detect(board)
dt = (time.perf_counter() - t0) * 1000

print(f"  Found {len(corners)} corners in {dt:.1f}ms")
if corners:
    print(f"  Top 5: {[(c.x, c.y, f'{c.score:.3f}') for c in corners[:5]]}")

# Draw corners
annotated = board.draw_keypoints([(c.x, c.y) for c in corners],
                                 color=(0, 255, 0), radius=4)
annotated.save("detect_corners.jpg")
print("  Saved detect_corners.jpg")


# ── 2. Lucas-Kanade Optical Flow ─────────────────────────────────────────────
print("\n─── Optical Flow (Lucas-Kanade) ───")
frame1 = make_motion_frame(ball_x=100, ball_y=120)
frame2 = make_motion_frame(ball_x=115, ball_y=125)  # ball moved

flow_tracker = OpticalFlow(win_size=15)

# Track corners from frame1 into frame2
pts = [(c.x, c.y) for c in corners[:50]]
if pts:
    t0 = time.perf_counter()
    vectors = flow_tracker.calc(frame1, frame2, pts)
    dt = (time.perf_counter() - t0) * 1000

    valid = [v for v in vectors if v.valid]
    moving = [v for v in valid if v.magnitude > 1.0]
    print(f"  Tracked {len(valid)} points in {dt:.1f}ms")
    print(f"  Moving points (mag>1): {len(moving)}")

    if moving:
        avg_u = sum(v.u for v in moving) / len(moving)
        avg_v = sum(v.v for v in moving) / len(moving)
        print(f"  Average motion: ({avg_u:+.1f}, {avg_v:+.1f}) pixels")

    # Draw flow
    vis = frame2.to_rgb()
    vis = flow_tracker.draw_flow(vis, vectors, color=(255, 255, 0), scale=5.0)
    vis.save("detect_flow.jpg")
    print("  Saved detect_flow.jpg")


# ── 3. Motion Detection ───────────────────────────────────────────────────────
print("\n─── Motion Detection ───")
motion_det = MotionDetector(threshold=25, min_area=200, learning_rate=0.05)

# Simulate camera frames
background = make_motion_frame(ball_x=160, ball_y=120, r=0)  # empty scene
frames_with_motion = [
    make_motion_frame(ball_x=80+i*10, ball_y=100, r=25)
    for i in range(5)
]

# Warm up background model
for _ in range(10):
    motion_det.detect(background)

detections_found = 0
for i, frame in enumerate(frames_with_motion):
    dets = motion_det.detect(frame)
    if dets:
        detections_found += 1
        d = dets[0]
        annotated = frame.draw_rect(d.x, d.y, d.w, d.h, (255,0,0), 2)
        annotated.save(f"detect_motion_{i}.jpg")

print(f"  Detected motion in {detections_found}/{len(frames_with_motion)} frames")
print("  Saved detect_motion_*.jpg")


# ── 4. Template Matching ──────────────────────────────────────────────────────
print("\n─── Template Matching ───")
# Create scene with a known pattern
scene = make_checkerboard(320, 240, 32)
# Crop a template from it
template = scene.crop(64, 64, 64, 64)  # 64x64 patch
template.save("detect_template.jpg")

matcher = TemplateMatcher(template, method="ncc", threshold=0.8)

t0 = time.perf_counter()
result = matcher.match(scene)
dt = (time.perf_counter() - t0) * 1000
print(f"  Template match in {dt:.1f}ms")
if result:
    print(f"  Found at ({result.x}, {result.y}) score={result.score:.3f}")
    annotated = scene.draw_rect(result.x, result.y,
                                template.width, template.height,
                                (255, 0, 255), 3)
    annotated.save("detect_match_result.jpg")
    print("  Saved detect_match_result.jpg")
else:
    print("  Not found (below threshold)")


# ── 5. Contour Detection ──────────────────────────────────────────────────────
print("\n─── Contour Detection ───")
# Create binary mask with shapes
arr = np.zeros((240, 320), dtype=np.uint8)
# Add rectangles
arr[40:80, 50:130] = 255
arr[100:160, 180:260] = 255
arr[170:210, 80:200] = 255

mask = from_numpy(arr)
mask.save("detect_mask.jpg")

analyzer = ContourAnalyzer(min_area=100, min_length=20)
contours = analyzer.find(mask)
print(f"  Found {len(contours)} contours")
for i, c in enumerate(contours):
    x, y, w, h = c.bbox
    print(f"    #{i+1}: bbox=({x},{y},{w},{h}) area={c.area:.0f} len={c.length}")

# Draw on color image
color_mask = from_numpy(np.stack([arr, np.zeros_like(arr),
                                  np.zeros_like(arr)], axis=-1))
color_mask = color_mask.draw_contours(
    [{"points": c.points} for c in contours], color=(0, 255, 0))
color_mask.save("detect_contours.jpg")
print("  Saved detect_contours.jpg")


# ── 6. HOG Descriptor (feature analysis) ─────────────────────────────────────
print("\n─── HOG Descriptor ───")
if ext:
    patch = board.to_gray().resize(64, 128)
    t0 = time.perf_counter()
    desc = ext.hog_descriptor(patch._data, cell_size=8, block_size=2, num_bins=9)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  HOG descriptor: {len(desc)} values in {dt:.1f}ms")
    print(f"  Mean: {sum(desc)/len(desc):.4f}, Max: {max(desc):.4f}")
else:
    from litecv._detection import HOGMatcher
    hog = HOGMatcher(board, win_w=64, win_h=64)
    print(f"  HOG ref descriptor: {len(hog._ref_hog)} values (Python)")


print("\n✓ All detection demos complete!")
print("  Output files: detect_*.jpg")
