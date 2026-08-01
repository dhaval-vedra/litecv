/*
 * lcv_simd.hpp — ARM NEON + x86 SSE/AVX auto-detect helpers
 * Provides SIMD-accelerated pixel operations for litecv-advanced.
 * Falls back to scalar if no SIMD available (always safe).
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>

// ── Detect SIMD capabilities ─────────────────────────────────────────────────
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define LCV_HAVE_NEON 1
#else
  #define LCV_HAVE_NEON 0
#endif

#if defined(__SSE2__)
  #include <emmintrin.h>
  #define LCV_HAVE_SSE2 1
#else
  #define LCV_HAVE_SSE2 0
#endif

#if defined(__AVX2__)
  #include <immintrin.h>
  #define LCV_HAVE_AVX2 1
#else
  #define LCV_HAVE_AVX2 0
#endif

namespace lcv_simd {

// ── Clamp helper ─────────────────────────────────────────────────────────────
inline uint8_t clamp_u8(int v) {
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

// ── Saturate-add (pixel + delta, clamped) ────────────────────────────────────
// Adds delta to each pixel in src, writes to dst. len = number of bytes.
inline void add_saturate(const uint8_t* src, uint8_t* dst, int len, int delta) {
#if LCV_HAVE_NEON
    int i = 0;
    if (delta >= 0) {
        uint8_t d = (uint8_t)(delta > 255 ? 255 : delta);
        uint8x16_t vd = vdupq_n_u8(d);
        for (; i + 16 <= len; i += 16) {
            uint8x16_t vs = vld1q_u8(src + i);
            vst1q_u8(dst + i, vqaddq_u8(vs, vd));
        }
    } else {
        uint8_t d = (uint8_t)((-delta) > 255 ? 255 : -delta);
        uint8x16_t vd = vdupq_n_u8(d);
        for (; i + 16 <= len; i += 16) {
            uint8x16_t vs = vld1q_u8(src + i);
            vst1q_u8(dst + i, vqsubq_u8(vs, vd));
        }
    }
    for (; i < len; i++) dst[i] = clamp_u8((int)src[i] + delta);
#elif LCV_HAVE_SSE2
    int i = 0;
    if (delta >= 0) {
        uint8_t d = (uint8_t)(delta > 255 ? 255 : delta);
        __m128i vd = _mm_set1_epi8((char)d);
        for (; i + 16 <= len; i += 16) {
            __m128i vs = _mm_loadu_si128((__m128i*)(src + i));
            _mm_storeu_si128((__m128i*)(dst + i), _mm_adds_epu8(vs, vd));
        }
    } else {
        uint8_t d = (uint8_t)((-delta) > 255 ? 255 : -delta);
        __m128i vd = _mm_set1_epi8((char)d);
        for (; i + 16 <= len; i += 16) {
            __m128i vs = _mm_loadu_si128((__m128i*)(src + i));
            _mm_storeu_si128((__m128i*)(dst + i), _mm_subs_epu8(vs, vd));
        }
    }
    for (; i < len; i++) dst[i] = clamp_u8((int)src[i] + delta);
#else
    for (int i = 0; i < len; i++) dst[i] = clamp_u8((int)src[i] + delta);
#endif
}

// ── Multiply pixels by float scale (brightness) ──────────────────────────────
inline void scale_pixels(const uint8_t* src, uint8_t* dst, int len, float scale) {
#if LCV_HAVE_NEON
    int i = 0;
    float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 8 <= len; i += 8) {
        uint8x8_t raw = vld1_u8(src + i);
        uint16x8_t u16 = vmovl_u8(raw);
        uint16x4_t lo = vget_low_u16(u16);
        uint16x4_t hi = vget_high_u16(u16);
        float32x4_t flo = vcvtq_f32_u32(vmovl_u16(lo));
        float32x4_t fhi = vcvtq_f32_u32(vmovl_u16(hi));
        flo = vmulq_f32(flo, vs);
        fhi = vmulq_f32(fhi, vs);
        uint32x4_t rlo = vcvtq_u32_f32(flo);
        uint32x4_t rhi = vcvtq_u32_f32(fhi);
        uint8x8_t result = vqmovn_u16(vcombine_u16(vqmovn_u32(rlo), vqmovn_u32(rhi)));
        vst1_u8(dst + i, result);
    }
    for (; i < len; i++) dst[i] = clamp_u8((int)((float)src[i] * scale));
#else
    for (int i = 0; i < len; i++) dst[i] = clamp_u8((int)((float)src[i] * scale));
#endif
}

// ── Grayscale (RGB → Y) ──────────────────────────────────────────────────────
// src: RGB interleaved, dst: single channel. len = number of pixels.
inline void rgb_to_gray(const uint8_t* src_rgb, uint8_t* dst_gray, int num_pixels) {
#if LCV_HAVE_NEON
    int i = 0;
    for (; i + 8 <= num_pixels; i += 8) {
        uint8x8x3_t rgb = vld3_u8(src_rgb + i * 3);
        // Y = 0.299R + 0.587G + 0.114B  ≈  (77R + 150G + 29B) >> 8
        uint16x8_t r16 = vmovl_u8(rgb.val[0]);
        uint16x8_t g16 = vmovl_u8(rgb.val[1]);
        uint16x8_t b16 = vmovl_u8(rgb.val[2]);
        uint16x8_t y = vaddq_u16(
            vaddq_u16(vshrq_n_u16(vmulq_n_u16(r16, 77), 8),
                      vshrq_n_u16(vmulq_n_u16(g16, 150), 8)),
            vshrq_n_u16(vmulq_n_u16(b16, 29), 8));
        vst1_u8(dst_gray + i, vmovn_u16(y));
    }
    for (; i < num_pixels; i++) {
        int r = src_rgb[i*3], g = src_rgb[i*3+1], b = src_rgb[i*3+2];
        dst_gray[i] = (uint8_t)((77*r + 150*g + 29*b) >> 8);
    }
#else
    for (int i = 0; i < num_pixels; i++) {
        int r = src_rgb[i*3], g = src_rgb[i*3+1], b = src_rgb[i*3+2];
        dst_gray[i] = (uint8_t)((77*r + 150*g + 29*b) >> 8);
    }
#endif
}

// ── Horizontal box-blur row accumulator ──────────────────────────────────────
// Fast sliding-window 1D blur along a row (uint8 → uint8 with radius r)
inline void hblur_row(const uint8_t* src, uint8_t* dst, int len, int r) {
    int window = 2 * r + 1;
    int sum = 0;
    for (int i = 0; i < window && i < len; i++) sum += src[i];
    for (int i = 0; i < len; i++) {
        dst[i] = (uint8_t)(sum / window);
        int add = (i + r + 1 < len) ? src[i + r + 1] : src[len - 1];
        int sub = (i - r > 0)       ? src[i - r - 1] : src[0];
        sum += add - sub;
    }
}

// ── Absolute difference ───────────────────────────────────────────────────────
inline void absdiff(const uint8_t* a, const uint8_t* b, uint8_t* dst, int len) {
#if LCV_HAVE_NEON
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        vst1q_u8(dst + i, vabdq_u8(va, vb));
    }
    for (; i < len; i++) dst[i] = (uint8_t)std::abs((int)a[i] - (int)b[i]);
#else
    for (int i = 0; i < len; i++) dst[i] = (uint8_t)std::abs((int)a[i] - (int)b[i]);
#endif
}

// ── Runtime SIMD capability query ─────────────────────────────────────────────
inline const char* simd_name() {
#if LCV_HAVE_NEON
    return "ARM NEON";
#elif LCV_HAVE_AVX2
    return "x86 AVX2";
#elif LCV_HAVE_SSE2
    return "x86 SSE2";
#else
    return "scalar";
#endif
}

} // namespace lcv_simd
