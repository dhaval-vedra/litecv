"""
pipeline_demo.py — litecv-advanced pipeline API demo

Shows all the ways to chain operations: operator style, Pipeline object,
preset pipelines, branching, benchmarking.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import litecv
from litecv import open_image, new_image, from_numpy, pipeline as P, Pipeline

# ── Create a test image (colorful gradient) ───────────────────────────────────
print("Creating test image…")
h, w = 480, 640
arr = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        arr[y, x] = [int(x/w*255), int(y/h*255), int((x+y)/(w+h)*255)]

# Add some shapes
img = from_numpy(arr)
img = img.draw_rect(50, 50, 200, 150, color=(255, 0, 0), thickness=3)
img = img.draw_circle(400, 200, 80, color=(0, 255, 0), thickness=3)
img = img.draw_text("litecv-advanced", 100, 400, color=(255, 255, 255))
img.save("demo_source.jpg")
print("Saved demo_source.jpg")

# ── Method 1: Fluent/method chaining ─────────────────────────────────────────
print("\n[1] Method chaining:")
result = (img
          .clahe(clip_limit=2.5)
          .unsharp_mask(amount=1.5)
          .sepia())
result.save("demo_sepia.jpg")
print("  → demo_sepia.jpg")

# ── Method 2: Operator | style ────────────────────────────────────────────────
print("\n[2] Operator style (|):")
result = (img
          | P.clahe()
          | P.thermal()
          | P.save("demo_thermal.jpg"))
print("  → demo_thermal.jpg")

# ── Method 3: Pipeline object (reusable) ──────────────────────────────────────
print("\n[3] Reusable Pipeline object:")
camera_pipeline = Pipeline([
    P.clahe(clip_limit=3.0),
    P.denoise(),
    P.unsharp_mask(amount=1.2),
    P.contrast(1.3),
])
result = camera_pipeline(img)
result.save("demo_enhanced.jpg")
print("  → demo_enhanced.jpg")

# Apply same pipeline to multiple images
images = [img.crop(0, 0, 200, 200), img.crop(200, 0, 200, 200), img.crop(0, 200, 200, 200)]
enhanced = camera_pipeline.map(images)
for i, e in enumerate(enhanced):
    e.save(f"demo_enhanced_tile_{i}.jpg")
print(f"  → {len(images)} tiles processed")

# ── Method 4: Pipeline composition (>>) ──────────────────────────────────────
print("\n[4] Pipeline composition (p1 >> p2):")
preprocess = Pipeline([P.clahe(), P.denoise()])
effect     = Pipeline([P.night_vision()])
full_pipe  = preprocess >> effect
result = full_pipe(img)
result.save("demo_night.jpg")
print("  → demo_night.jpg")

# ── Method 5: Branching (save multiple versions) ──────────────────────────────
print("\n[5] Branching:")
result = (img
          | P.clahe()
          | P.branch(
              P.save("demo_branch_sepia.jpg") if False else P.tap(lambda x: None),  # noqa
              P.save("demo_branch_enhanced.jpg") if False else P.tap(lambda x: None)
            )
          | P.sepia())
result.save("demo_branch_final.jpg")
print("  → demo_branch_final.jpg")

# ── Method 6: Preset pipelines ────────────────────────────────────────────────
print("\n[6] Preset pipelines:")
from litecv.pipeline import enhance_camera, night_mode, document_scan

presets = {
    "enhance_camera":  enhance_camera(),
    "night_mode":      night_mode(),
    "document_scan":   document_scan(),
}
for name, preset in presets.items():
    result = preset(img)
    result.save(f"demo_preset_{name}.jpg")
    print(f"  → demo_preset_{name}.jpg")

# ── Method 7: Benchmark ───────────────────────────────────────────────────────
print("\n[7] Benchmark (10 iterations):")
stats = camera_pipeline.benchmark(img, n=10)
print(f"  Mean: {stats['mean_ms']:.1f}ms | Min: {stats['min_ms']:.1f}ms | FPS: {stats['fps']:.1f}")

print("\n✓ All pipeline demos complete!")
