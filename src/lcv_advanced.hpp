/*
 * lcv_advanced.hpp — Advanced vision algorithms for edge devices
 *
 * Features (all pure C++, no OpenCV):
 *   - CLAHE  (Contrast Limited Adaptive Histogram Equalization)
 *   - Bilateral filter (edge-preserving blur)
 *   - Guided filter (edge-preserving, faster than bilateral)
 *   - Harris corner detector
 *   - Lucas-Kanade sparse optical flow
 *   - MOG2-simplified background subtractor
 *   - HOG descriptor + sliding window detector
 *   - Contour detection (simplified border-following)
 *   - Template matching (SSD / NCC)
 */
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <cstring>
#include "lcv_simd.hpp"

namespace lcv_adv {

using u8  = uint8_t;
using f32 = float;
using i32 = int32_t;

// ═══════════════════════════════════════════════════════════════════════════
//  CLAHE — Contrast Limited Adaptive Histogram Equalization
// ═══════════════════════════════════════════════════════════════════════════
/*
 * Works on grayscale images. Divides image into tiles, builds a
 * histogram per tile, clips it at clip_limit, redistributes clipped
 * counts, interpolates LUT bilinearly across tiles.
 */
inline void clahe(const u8* src, u8* dst, int w, int h,
                  int tile_w = 8, int tile_h = 8, float clip_limit = 2.0f) {
    int tw = (w + tile_w - 1) / tile_w; // number of tiles horizontally
    int th = (h + tile_h - 1) / tile_h; // number of tiles vertically

    // Build per-tile LUTs
    std::vector<u8> luts(tw * th * 256);

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            // Tile region (clamped)
            int x0 = tx * tile_w, y0 = ty * tile_h;
            int x1 = std::min(x0 + tile_w, w);
            int y1 = std::min(y0 + tile_h, h);
            int area = (x1 - x0) * (y1 - y0);

            // Histogram
            int hist[256] = {};
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                    hist[src[y * w + x]]++;

            // Clip
            int clip = (int)(clip_limit * area / 256.0f);
            clip = std::max(clip, 1);
            int excess = 0;
            for (int i = 0; i < 256; i++) {
                if (hist[i] > clip) { excess += hist[i] - clip; hist[i] = clip; }
            }
            // Redistribute excess uniformly
            int add_each = excess / 256;
            int remainder = excess % 256;
            for (int i = 0; i < 256; i++) hist[i] += add_each;
            for (int i = 0; i < remainder; i++) hist[i]++;

            // CDF → LUT
            u8* lut = &luts[(ty * tw + tx) * 256];
            int cdf = 0, cdf_min = -1;
            for (int i = 0; i < 256; i++) {
                cdf += hist[i];
                if (cdf_min < 0 && cdf > 0) cdf_min = cdf;
                int v = (area > cdf_min) ?
                    (int)std::round(255.0f * (cdf - cdf_min) / (area - cdf_min)) : 0;
                lut[i] = (u8)std::min(255, std::max(0, v));
            }
        }
    }

    // Bilinear interpolation between tile LUTs
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            u8 pix = src[y * w + x];

            // Which tile the pixel belongs to (continuous coords)
            float tx_f = (float)x / tile_w - 0.5f;
            float ty_f = (float)y / tile_h - 0.5f;
            int tx0 = (int)std::floor(tx_f), ty0 = (int)std::floor(ty_f);
            int tx1 = tx0 + 1, ty1 = ty0 + 1;
            float ax = tx_f - tx0, ay = ty_f - ty0;

            // Clamp tile indices
            tx0 = std::max(0, std::min(tw-1, tx0));
            tx1 = std::max(0, std::min(tw-1, tx1));
            ty0 = std::max(0, std::min(th-1, ty0));
            ty1 = std::max(0, std::min(th-1, ty1));

            float v00 = luts[(ty0*tw + tx0)*256 + pix];
            float v01 = luts[(ty0*tw + tx1)*256 + pix];
            float v10 = luts[(ty1*tw + tx0)*256 + pix];
            float v11 = luts[(ty1*tw + tx1)*256 + pix];

            float val = v00*(1-ax)*(1-ay) + v01*ax*(1-ay)
                      + v10*(1-ax)*ay     + v11*ax*ay;
            dst[y * w + x] = (u8)std::min(255.f, std::max(0.f, val));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Bilateral Filter (edge-preserving smoothing)
// ═══════════════════════════════════════════════════════════════════════════
inline void bilateral_filter(const u8* src, u8* dst, int w, int h,
                             int d, float sigma_color, float sigma_space) {
    int r = d / 2;
    float inv_color2 = 1.0f / (2.0f * sigma_color * sigma_color);
    float inv_space2 = 1.0f / (2.0f * sigma_space * sigma_space);

    // Precompute space weights
    std::vector<float> space_w((2*r+1)*(2*r+1));
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            space_w[(dy+r)*(2*r+1)+(dx+r)] =
                std::exp(-(dy*dy + dx*dx) * inv_space2);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float center = src[y * w + x];
            float sum = 0, wsum = 0;
            for (int dy = -r; dy <= r; dy++) {
                int ny = std::max(0, std::min(h-1, y + dy));
                for (int dx = -r; dx <= r; dx++) {
                    int nx = std::max(0, std::min(w-1, x + dx));
                    float neighbor = src[ny * w + nx];
                    float diff = neighbor - center;
                    float w_color = std::exp(-diff * diff * inv_color2);
                    float w_space = space_w[(dy+r)*(2*r+1)+(dx+r)];
                    float wt = w_color * w_space;
                    sum  += neighbor * wt;
                    wsum += wt;
                }
            }
            dst[y * w + x] = (u8)(sum / wsum + 0.5f);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Harris Corner Detector
// ═══════════════════════════════════════════════════════════════════════════
struct Corner { int x, y; float score; };

inline std::vector<Corner> harris_corners(
    const u8* gray, int w, int h,
    int block_size = 3, int ksize = 3,
    float k = 0.04f, float threshold = 0.01f,
    int max_corners = 500)
{
    int n = w * h;
    std::vector<float> Ix(n), Iy(n), Ix2(n), Iy2(n), Ixy(n), R(n, 0.f);

    // Sobel gradients
    int ks = ksize / 2;
    for (int y = ks; y < h - ks; y++) {
        for (int x = ks; x < w - ks; x++) {
            float gx = -(float)gray[(y-1)*w+(x-1)] + gray[(y-1)*w+(x+1)]
                       - 2.f*gray[y*w+(x-1)]        + 2.f*gray[y*w+(x+1)]
                       - gray[(y+1)*w+(x-1)]         + gray[(y+1)*w+(x+1)];
            float gy = -(float)gray[(y-1)*w+(x-1)] - 2.f*gray[(y-1)*w+x] - gray[(y-1)*w+(x+1)]
                       + gray[(y+1)*w+(x-1)]         + 2.f*gray[(y+1)*w+x] + gray[(y+1)*w+(x+1)];
            Ix2[y*w+x] = gx * gx;
            Iy2[y*w+x] = gy * gy;
            Ixy[y*w+x] = gx * gy;
        }
    }

    // Box-filter the gradient products (block_size neighborhood)
    int br = block_size / 2;
    for (int y = br; y < h - br; y++) {
        for (int x = br; x < w - br; x++) {
            float A = 0, B = 0, C = 0;
            for (int dy = -br; dy <= br; dy++)
                for (int dx = -br; dx <= br; dx++) {
                    A += Ix2[(y+dy)*w+(x+dx)];
                    B += Iy2[(y+dy)*w+(x+dx)];
                    C += Ixy[(y+dy)*w+(x+dx)];
                }
            float det   = A * B - C * C;
            float trace = A + B;
            R[y*w+x] = det - k * trace * trace;
        }
    }

    // Normalize R
    float maxR = *std::max_element(R.begin(), R.end());
    if (maxR <= 0) return {};

    // Non-max suppression + threshold
    std::vector<Corner> corners;
    int nr = 5; // suppression radius
    for (int y = nr; y < h - nr; y++) {
        for (int x = nr; x < w - nr; x++) {
            float r = R[y*w+x] / maxR;
            if (r < threshold) continue;
            // Check local max
            bool local_max = true;
            for (int dy = -nr; dy <= nr && local_max; dy++)
                for (int dx = -nr; dx <= nr && local_max; dx++)
                    if (R[(y+dy)*w+(x+dx)] > R[y*w+x]) local_max = false;
            if (local_max) corners.push_back({x, y, r});
        }
    }

    // Sort by score descending, take top N
    std::sort(corners.begin(), corners.end(),
              [](const Corner& a, const Corner& b){ return a.score > b.score; });
    if ((int)corners.size() > max_corners)
        corners.resize(max_corners);
    return corners;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Lucas-Kanade Sparse Optical Flow
// ═══════════════════════════════════════════════════════════════════════════
struct FlowPoint { float x, y, u, v; bool valid; };

inline std::vector<FlowPoint> lk_optical_flow(
    const u8* prev, const u8* next,
    const std::vector<std::pair<float,float>>& pts,
    int w, int h, int win_size = 15, int max_iter = 30, float eps = 0.01f)
{
    std::vector<FlowPoint> result;
    int wr = win_size / 2;

    auto get_px = [&](const u8* img, float x, float y) -> float {
        int x0 = (int)x, y0 = (int)y;
        x0 = std::max(0, std::min(w-2, x0));
        y0 = std::max(0, std::min(h-2, y0));
        float fx = x - x0, fy = y - y0;
        return (1-fx)*(1-fy)*img[y0*w+x0]     + fx*(1-fy)*img[y0*w+x0+1]
             + (1-fx)*fy*img[(y0+1)*w+x0]      + fx*fy*img[(y0+1)*w+x0+1];
    };

    for (auto& [px, py] : pts) {
        // Spatial gradients (from prev)
        float A11=0, A12=0, A22=0;
        std::vector<float> Ix_w, Iy_w;
        for (int dy = -wr; dy <= wr; dy++) {
            for (int dx = -wr; dx <= wr; dx++) {
                float gx = get_px(prev, px+dx+1, py+dy) - get_px(prev, px+dx-1, py+dy);
                float gy = get_px(prev, px+dx, py+dy+1) - get_px(prev, px+dx, py+dy-1);
                gx *= 0.5f; gy *= 0.5f;
                A11 += gx*gx; A12 += gx*gy; A22 += gy*gy;
                Ix_w.push_back(gx); Iy_w.push_back(gy);
            }
        }
        float det = A11*A22 - A12*A12;
        if (std::abs(det) < 1e-6f) {
            result.push_back({px, py, 0, 0, false});
            continue;
        }
        float inv_det = 1.0f / det;

        float u = 0, v = 0;
        for (int iter = 0; iter < max_iter; iter++) {
            float b1=0, b2=0;
            int idx = 0;
            for (int dy = -wr; dy <= wr; dy++) {
                for (int dx = -wr; dx <= wr; dx++) {
                    float It = get_px(next, px+dx+u, py+dy+v) - get_px(prev, px+dx, py+dy);
                    b1 -= Ix_w[idx] * It;
                    b2 -= Iy_w[idx] * It;
                    idx++;
                }
            }
            float du = inv_det * (A22*b1 - A12*b2);
            float dv = inv_det * (A11*b2 - A12*b1);
            u += du; v += dv;
            if (du*du + dv*dv < eps*eps) break;
        }
        result.push_back({px, py, u, v, true});
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Background Subtractor (running average + threshold)
// ═══════════════════════════════════════════════════════════════════════════
class BackgroundSubtractor {
public:
    std::vector<float> bg_model;
    int w = 0, h = 0;
    float learning_rate;
    float threshold;

    BackgroundSubtractor(float lr = 0.05f, float thr = 25.0f)
        : learning_rate(lr), threshold(thr) {}

    // gray: grayscale input, mask_out: foreground mask (255=FG, 0=BG)
    void apply(const u8* gray, u8* mask_out, int width, int height) {
        if (w != width || h != height) {
            w = width; h = height;
            bg_model.assign(w * h, 0.f);
            for (int i = 0; i < w*h; i++) bg_model[i] = gray[i];
        }
        for (int i = 0; i < w*h; i++) {
            float diff = std::abs((float)gray[i] - bg_model[i]);
            mask_out[i] = (diff > threshold) ? 255 : 0;
            bg_model[i] = (1.f - learning_rate) * bg_model[i] + learning_rate * gray[i];
        }
    }

    void reset() { bg_model.clear(); w = 0; h = 0; }
};

// ═══════════════════════════════════════════════════════════════════════════
//  HOG Descriptor (Histogram of Oriented Gradients)
// ═══════════════════════════════════════════════════════════════════════════
struct HOGParams {
    int cell_size  = 8;   // pixels per cell
    int block_size = 2;   // cells per block (block_size x block_size)
    int num_bins   = 9;   // orientation bins (0°-180°)
};

inline std::vector<float> compute_hog(
    const u8* gray, int w, int h,
    const HOGParams& params = {})
{
    int cs = params.cell_size;
    int bs = params.block_size;
    int nb = params.num_bins;
    int cells_x = w / cs, cells_y = h / cs;

    // Per-cell histograms
    std::vector<float> cell_hist(cells_y * cells_x * nb, 0.f);

    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            float* hist = &cell_hist[(cy * cells_x + cx) * nb];
            for (int dy = 0; dy < cs; dy++) {
                for (int dx = 0; dx < cs; dx++) {
                    int y = cy*cs + dy, x = cx*cs + dx;
                    if (y == 0 || y >= h-1 || x == 0 || x >= w-1) continue;
                    float gx = (float)gray[y*w+x+1] - gray[y*w+x-1];
                    float gy = (float)gray[(y+1)*w+x] - gray[(y-1)*w+x];
                    float mag = std::sqrt(gx*gx + gy*gy);
                    float ang = std::atan2(gy, gx) * (180.f / M_PI);
                    if (ang < 0) ang += 180.f;
                    ang = std::fmod(ang, 180.f);
                    int bin = (int)(ang / 180.f * nb) % nb;
                    hist[bin] += mag;
                }
            }
        }
    }

    // Block normalization (L2-Hys)
    std::vector<float> descriptor;
    descriptor.reserve((cells_y-bs+1) * (cells_x-bs+1) * bs*bs*nb);

    for (int by = 0; by <= cells_y - bs; by++) {
        for (int bx = 0; bx <= cells_x - bs; bx++) {
            float block[4*9] = {};
            int idx = 0;
            for (int cy2 = by; cy2 < by+bs; cy2++)
                for (int cx2 = bx; cx2 < bx+bs; cx2++)
                    for (int b = 0; b < nb; b++)
                        block[idx++] = cell_hist[(cy2*cells_x+cx2)*nb+b];
            // L2 norm
            float norm2 = 0;
            for (int i = 0; i < idx; i++) norm2 += block[i]*block[i];
            float inv_norm = 1.0f / std::sqrt(norm2 + 1e-6f);
            for (int i = 0; i < idx; i++) {
                float v = block[i] * inv_norm;
                // Hys clamp
                v = std::min(v, 0.2f);
                descriptor.push_back(v);
            }
        }
    }
    return descriptor;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Template Matching (SSD / Normalized Cross-Correlation)
// ═══════════════════════════════════════════════════════════════════════════
enum class MatchMethod { SSD, NCC };

struct MatchResult { int x, y; float score; };

inline MatchResult template_match(
    const u8* img, int iw, int ih,
    const u8* tmpl, int tw, int th,
    MatchMethod method = MatchMethod::NCC)
{
    MatchResult best = {0, 0, (method == MatchMethod::SSD) ? 1e30f : -1.f};

    // Precompute template mean for NCC
    float t_mean = 0;
    int tn = tw * th;
    if (method == MatchMethod::NCC) {
        for (int i = 0; i < tn; i++) t_mean += tmpl[i];
        t_mean /= tn;
    }

    for (int y = 0; y <= ih - th; y++) {
        for (int x = 0; x <= iw - tw; x++) {
            if (method == MatchMethod::SSD) {
                float ssd = 0;
                for (int ty = 0; ty < th; ty++)
                    for (int tx = 0; tx < tw; tx++) {
                        float d = (float)img[(y+ty)*iw+(x+tx)] - tmpl[ty*tw+tx];
                        ssd += d * d;
                    }
                if (ssd < best.score) best = {x, y, ssd};
            } else {
                // NCC
                float i_mean = 0;
                for (int ty = 0; ty < th; ty++)
                    for (int tx = 0; tx < tw; tx++)
                        i_mean += img[(y+ty)*iw+(x+tx)];
                i_mean /= tn;
                float num = 0, di = 0, dt = 0;
                for (int ty = 0; ty < th; ty++)
                    for (int tx = 0; tx < tw; tx++) {
                        float fi = (float)img[(y+ty)*iw+(x+tx)] - i_mean;
                        float ft = (float)tmpl[ty*tw+tx] - t_mean;
                        num += fi * ft;
                        di  += fi * fi;
                        dt  += ft * ft;
                    }
                float denom = std::sqrt(di * dt) + 1e-10f;
                float ncc = num / denom;
                if (ncc > best.score) best = {x, y, ncc};
            }
        }
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Simple Contour Detection (connected-component border)
// ═══════════════════════════════════════════════════════════════════════════
using Contour = std::vector<std::pair<int,int>>;

// Extract contours from a binary mask (255=FG). Returns list of contours.
inline std::vector<Contour> find_contours(const u8* mask, int w, int h, int min_len = 10) {
    std::vector<bool> visited(w * h, false);
    std::vector<Contour> contours;

    auto is_border = [&](int x, int y) -> bool {
        if (mask[y*w+x] == 0) return false;
        // Check 4-neighbors
        if (x == 0 || x == w-1 || y == 0 || y == h-1) return true;
        return mask[y*w+x-1] == 0 || mask[y*w+x+1] == 0 ||
               mask[(y-1)*w+x] == 0 || mask[(y+1)*w+x] == 0;
    };

    int dx8[] = {1,1,0,-1,-1,-1,0,1};
    int dy8[] = {0,1,1, 1, 0,-1,-1,-1};

    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            if (visited[y*w+x] || !is_border(x, y)) continue;
            // BFS border trace
            Contour contour;
            std::vector<std::pair<int,int>> queue;
            queue.push_back({x, y});
            visited[y*w+x] = true;
            while (!queue.empty()) {
                auto [cx, cy] = queue.back(); queue.pop_back();
                contour.push_back({cx, cy});
                for (int d = 0; d < 8; d++) {
                    int nx = cx + dx8[d], ny = cy + dy8[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    if (visited[ny*w+nx] || !is_border(nx, ny)) continue;
                    visited[ny*w+nx] = true;
                    queue.push_back({nx, ny});
                }
            }
            if ((int)contour.size() >= min_len)
                contours.push_back(std::move(contour));
        }
    }
    return contours;
}

// Bounding box of a contour
inline std::tuple<int,int,int,int> contour_bbox(const Contour& c) {
    int xmin=1<<30, xmax=-1, ymin=1<<30, ymax=-1;
    for (auto [x,y] : c) {
        xmin = std::min(xmin, x); xmax = std::max(xmax, x);
        ymin = std::min(ymin, y); ymax = std::max(ymax, y);
    }
    return {xmin, ymin, xmax-xmin+1, ymax-ymin+1};
}

// Contour area (shoelace)
inline float contour_area(const Contour& c) {
    if (c.size() < 3) return 0.f;
    float area = 0;
    int n = c.size();
    for (int i = 0; i < n; i++) {
        auto [x1,y1] = c[i];
        auto [x2,y2] = c[(i+1)%n];
        area += x1*y2 - x2*y1;
    }
    return std::abs(area) * 0.5f;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Image Sharpening (unsharp mask)
// ═══════════════════════════════════════════════════════════════════════════
inline void unsharp_mask(const u8* src, u8* dst, int w, int h, int ch,
                          float amount = 1.5f, int radius = 3) {
    int n = w * h * ch;
    std::vector<u8> blurred(n);
    // Simple box blur
    std::vector<u8> tmp(n);
    // Horizontal pass
    for (int y = 0; y < h; y++)
        for (int c2 = 0; c2 < ch; c2++)
            lcv_simd::hblur_row(src + y*w*ch + c2, tmp.data() + y*w*ch + c2,
                                w * ch, radius);
    // This is a simplified approach — apply per channel
    for (int i = 0; i < n; i++) {
        float sharp = (float)src[i] + amount * ((float)src[i] - (float)tmp[i]);
        dst[i] = lcv_simd::clamp_u8((int)sharp);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Histogram computation + equalization
// ═══════════════════════════════════════════════════════════════════════════
inline std::vector<int> compute_histogram(const u8* gray, int n) {
    std::vector<int> hist(256, 0);
    for (int i = 0; i < n; i++) hist[gray[i]]++;
    return hist;
}

inline void equalize_histogram(const u8* src, u8* dst, int n) {
    auto hist = compute_histogram(src, n);
    // CDF
    std::vector<u8> lut(256);
    int cdf = 0, cdf_min = -1;
    for (int i = 0; i < 256; i++) {
        cdf += hist[i];
        if (cdf_min < 0 && cdf > 0) cdf_min = cdf;
        int v = (n > cdf_min) ?
            (int)std::round(255.f * (cdf - cdf_min) / (float)(n - cdf_min)) : 0;
        lut[i] = (u8)std::min(255, std::max(0, v));
    }
    for (int i = 0; i < n; i++) dst[i] = lut[src[i]];
}

// ═══════════════════════════════════════════════════════════════════════════
//  Color analysis — dominant colors via simple quantization
// ═══════════════════════════════════════════════════════════════════════════
struct ColorInfo {
    uint8_t r, g, b;
    float   fraction;
};

inline std::vector<ColorInfo> dominant_colors(
    const u8* rgb, int num_pixels, int num_colors = 5, int quantize_bits = 5)
{
    // Quantize each channel to (256 >> quantize_bits) levels
    int q = 1 << quantize_bits;
    int levels = 256 / q;  // bins per channel
    int total_bins = levels * levels * levels;
    std::vector<int> counts(total_bins, 0);
    std::vector<long long> sum_r(total_bins, 0), sum_g(total_bins, 0), sum_b(total_bins, 0);

    for (int i = 0; i < num_pixels; i++) {
        int r = rgb[i*3+0] / q;
        int g = rgb[i*3+1] / q;
        int b = rgb[i*3+2] / q;
        int idx = r * levels * levels + g * levels + b;
        counts[idx]++;
        sum_r[idx] += rgb[i*3+0];
        sum_g[idx] += rgb[i*3+1];
        sum_b[idx] += rgb[i*3+2];
    }

    // Sort bins by count
    std::vector<int> order(total_bins);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + num_colors, order.end(),
                      [&](int a, int b2){ return counts[a] > counts[b2]; });

    std::vector<ColorInfo> result;
    for (int i = 0; i < num_colors && i < (int)order.size(); i++) {
        int idx = order[i];
        if (counts[idx] == 0) break;
        ColorInfo ci;
        ci.r = (u8)(sum_r[idx] / counts[idx]);
        ci.g = (u8)(sum_g[idx] / counts[idx]);
        ci.b = (u8)(sum_b[idx] / counts[idx]);
        ci.fraction = (float)counts[idx] / num_pixels;
        result.push_back(ci);
    }
    return result;
}

} // namespace lcv_adv
