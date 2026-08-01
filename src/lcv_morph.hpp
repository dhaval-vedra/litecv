/*
 * lcv_morph.hpp — Morphological operations (erode, dilate, open, close, gradient)
 * Pure C++, no dependencies. Works on grayscale uint8 images.
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

namespace lcv_morph {

using u8 = uint8_t;

// ── Structuring element shapes ─────────────────────────────────────────────
enum class MorphShape { RECT, ELLIPSE, CROSS };

// Build a flat structuring element mask (ksize x ksize)
inline std::vector<bool> make_kernel(int ksize, MorphShape shape) {
    int r = ksize / 2;
    std::vector<bool> mask(ksize * ksize, false);
    for (int y = 0; y < ksize; y++) {
        for (int x = 0; x < ksize; x++) {
            int dy = y - r, dx = x - r;
            switch (shape) {
                case MorphShape::RECT:
                    mask[y * ksize + x] = true;
                    break;
                case MorphShape::ELLIPSE: {
                    float rx = r > 0 ? r : 1, ry = r > 0 ? r : 1;
                    mask[y * ksize + x] = ((float)dx/rx)*dx/rx + ((float)dy/ry)*dy/ry <= 1.0f;
                    break;
                }
                case MorphShape::CROSS:
                    mask[y * ksize + x] = (dy == 0 || dx == 0);
                    break;
            }
        }
    }
    return mask;
}

// ── Erode (min filter with structuring element) ───────────────────────────
inline void erode(const u8* src, u8* dst, int w, int h, int ksize,
                  MorphShape shape = MorphShape::RECT) {
    int r = ksize / 2;
    auto mask = make_kernel(ksize, shape);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int minv = 255;
            for (int ky = 0; ky < ksize; ky++) {
                for (int kx = 0; kx < ksize; kx++) {
                    if (!mask[ky * ksize + kx]) continue;
                    int sy = std::max(0, std::min(h-1, y + ky - r));
                    int sx = std::max(0, std::min(w-1, x + kx - r));
                    minv = std::min(minv, (int)src[sy * w + sx]);
                }
            }
            dst[y * w + x] = (u8)minv;
        }
    }
}

// ── Dilate (max filter with structuring element) ──────────────────────────
inline void dilate(const u8* src, u8* dst, int w, int h, int ksize,
                   MorphShape shape = MorphShape::RECT) {
    int r = ksize / 2;
    auto mask = make_kernel(ksize, shape);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int maxv = 0;
            for (int ky = 0; ky < ksize; ky++) {
                for (int kx = 0; kx < ksize; kx++) {
                    if (!mask[ky * ksize + kx]) continue;
                    int sy = std::max(0, std::min(h-1, y + ky - r));
                    int sx = std::max(0, std::min(w-1, x + kx - r));
                    maxv = std::max(maxv, (int)src[sy * w + sx]);
                }
            }
            dst[y * w + x] = (u8)maxv;
        }
    }
}

// ── Open = erode then dilate (removes small noise) ────────────────────────
inline void morph_open(const u8* src, u8* dst, u8* tmp, int w, int h, int ksize,
                        MorphShape shape = MorphShape::RECT) {
    erode(src, tmp, w, h, ksize, shape);
    dilate(tmp, dst, w, h, ksize, shape);
}

// ── Close = dilate then erode (fills small holes) ─────────────────────────
inline void morph_close(const u8* src, u8* dst, u8* tmp, int w, int h, int ksize,
                         MorphShape shape = MorphShape::RECT) {
    dilate(src, tmp, w, h, ksize, shape);
    erode(tmp, dst, w, h, ksize, shape);
}

// ── Morphological gradient = dilate - erode (edge outline) ────────────────
inline void morph_gradient(const u8* src, u8* dst, u8* d, u8* e, int w, int h,
                            int ksize, MorphShape shape = MorphShape::RECT) {
    dilate(src, d, w, h, ksize, shape);
    erode(src, e, w, h, ksize, shape);
    int n = w * h;
    for (int i = 0; i < n; i++)
        dst[i] = (u8)std::max(0, (int)d[i] - (int)e[i]);
}

// ── Top-hat = src - open (bright details on dark background) ─────────────
inline void tophat(const u8* src, u8* dst, u8* tmp1, u8* tmp2,
                   int w, int h, int ksize, MorphShape shape = MorphShape::RECT) {
    morph_open(src, tmp1, tmp2, w, h, ksize, shape);
    int n = w * h;
    for (int i = 0; i < n; i++)
        dst[i] = (u8)std::max(0, (int)src[i] - (int)tmp1[i]);
}

// ── Black-hat = close - src (dark details on bright background) ───────────
inline void blackhat(const u8* src, u8* dst, u8* tmp1, u8* tmp2,
                     int w, int h, int ksize, MorphShape shape = MorphShape::RECT) {
    morph_close(src, tmp1, tmp2, w, h, ksize, shape);
    int n = w * h;
    for (int i = 0; i < n; i++)
        dst[i] = (u8)std::max(0, (int)tmp1[i] - (int)src[i]);
}

} // namespace lcv_morph
