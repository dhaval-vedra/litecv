"""
basic_image.py — Core LiteImage API demo

All basic operations: create, geometry, color, adjustments,
filters, effects, drawing, analysis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import litecv
from litecv import from_numpy, new_image, LiteImage

print(litecv.backend_info())
print()

# ── Create images ─────────────────────────────────────────────────────────────
print("=== Create ===")
blank = new_image(400, 300, color=(30, 60, 90))
gradient = from_numpy(np.stack([
    np.tile(np.arange(400, dtype=np.uint8), (300, 1)),
    np.tile(np.arange(300, dtype=np.uint8)[:,np.newaxis], (1, 400)),
    np.full((300, 400), 128, dtype=np.uint8)
], axis=-1))
print(f"  Blank: {blank}")
print(f"  Gradient: {gradient}")

# ── Geometry ──────────────────────────────────────────────────────────────────
print("\n=== Geometry ===")
resized = gradient.resize(200, 150)
scaled  = gradient.scale(0.5)
cropped = gradient.crop(50, 50, 200, 150)
padded  = gradient.pad(20, 20, 20, 20, value=0)
rot90   = gradient.rotate90(1)
flipped = gradient.flip()
print(f"  Resize(200,150): {resized}")
print(f"  Scale(0.5):      {scaled}")
print(f"  Crop(50,50,200,150): {cropped}")
print(f"  Pad(20): {padded}")
print(f"  Rotate90: {rot90}")

# ── Color operations ──────────────────────────────────────────────────────────
print("\n=== Color ===")
gray = gradient.to_gray()
rgb  = gray.to_rgb()
r, g, b = gradient.split_channels()
merged  = LiteImage.merge_channels(r, g, b)
print(f"  Gray:    {gray}")
print(f"  RGB:     {rgb}")
print(f"  Split R: {r}")

# ── Pixel adjustments ─────────────────────────────────────────────────────────
print("\n=== Adjustments ===")
bright    = gradient.brightness(50)
dark      = gradient.brightness(-50)
high_con  = gradient.contrast(1.8)
low_con   = gradient.contrast(0.5)
gamma_adj = gradient.gamma(0.5)
inverted  = gradient.invert()
norm      = gradient.normalize()
thresh    = gray.threshold(128)
otsu      = gray.threshold_otsu()
print(f"  Brightness +50: mean={bright.mean_color()}")
print(f"  Contrast 1.8:   mean={high_con.mean_color()}")
print(f"  Gamma 0.5:      mean={gamma_adj.mean_color()}")
print(f"  Threshold:      {thresh}")
print(f"  Otsu:           {otsu}")

# ── Filters ───────────────────────────────────────────────────────────────────
print("\n=== Filters ===")
filters = {
    "blur(3)":        gradient.blur(3),
    "median(5)":      gradient.median(5),
    "bilateral":      gradient.bilateral(9, 75, 75),
    "sharpen":        gradient.sharpen(),
    "unsharp_mask":   gradient.unsharp_mask(1.5),
    "edges":          gradient.edges(),
    "laplacian":      gradient.laplacian(),
    "emboss":         gradient.emboss(),
    "denoise":        gradient.denoise(),
    "clahe":          gradient.clahe(clip_limit=3.0),
    "equalize":       gray.equalize(),
}
for name, result in filters.items():
    print(f"  {name:20s}: {result}")

# ── Morphological ops ─────────────────────────────────────────────────────────
print("\n=== Morphological ===")
# Create binary mask
mask = gray.threshold(128)
morph_ops = ["erode","dilate","open","close","gradient","tophat","blackhat"]
for op in morph_ops:
    result = mask.morph(op, ksize=3)
    print(f"  {op:12s}: {result}")

# ── Effects ───────────────────────────────────────────────────────────────────
print("\n=== Effects ===")
effects = {
    "sepia":       gradient.sepia(),
    "thermal":     gradient.thermal(),
    "night_vision":gradient.night_vision(),
    "sketch":      gradient.sketch(),
    "cartoon":     gradient.cartoon(),
    "edge_glow":   gradient.edge_glow(color=(0, 255, 120)),
}
for name, result in effects.items():
    print(f"  {name:15s}: {result}")

# ── Drawing ───────────────────────────────────────────────────────────────────
print("\n=== Drawing ===")
canvas = new_image(400, 300, color=(20, 20, 40))
canvas = canvas.draw_rect(50, 50, 200, 100, (255, 0, 0), 3)
canvas = canvas.draw_circle(300, 150, 60, (0, 255, 0), 2)
canvas = canvas.draw_line(0, 0, 400, 300, (255, 255, 0), 2)
canvas = canvas.draw_text("litecv-advanced", 20, 260, (255, 255, 255))
canvas.save("basic_canvas.jpg")
print("  Saved basic_canvas.jpg")

# ── Analysis ──────────────────────────────────────────────────────────────────
print("\n=== Analysis ===")
hist = gray.histogram()
print(f"  Histogram: {len(hist)} bins, max_count={max(hist)}")
colors = gradient.dominant_colors(num_colors=5)
print(f"  Dominant colors ({len(colors)}):")
for c in colors:
    print(f"    #{c['hex']} ({c['r']},{c['g']},{c['b']}) — {c['fraction']*100:.1f}%")
mean = gradient.mean_color()
std  = gradient.std_color()
print(f"  Mean color: R={mean[0]:.1f} G={mean[1]:.1f} B={mean[2]:.1f}")
print(f"  Std  color: R={std[0]:.1f} G={std[1]:.1f} B={std[2]:.1f}")

# ── AbsDiff ───────────────────────────────────────────────────────────────────
print("\n=== AbsDiff ===")
diff = bright.absdiff(dark)
print(f"  |bright - dark| mean: {diff.mean_color()}")

# ── Save all effects ──────────────────────────────────────────────────────────
print("\n=== Saving effect gallery ===")
for name, img in effects.items():
    img.save(f"basic_effect_{name}.jpg")
    print(f"  basic_effect_{name}.jpg")

print("\n✓ Basic image demo complete!")
