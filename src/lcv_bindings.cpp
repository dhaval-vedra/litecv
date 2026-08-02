/*
 * lcv_bindings.cpp — pybind11 extension module for litecv-advanced
 *
 * Exposes to Python:
 *   - Color conversions (RGB↔Gray↔HSV↔YCbCr↔Lab)
 *   - Image I/O (imread/imwrite/imdecode/imencode)
 *   - Filters (blur, median, bilateral, unsharp mask)
 *   - Morphological ops (erode/dilate/open/close/gradient/tophat/blackhat)
 *   - Resize / rotate / flip / transpose
 *   - Effects (sepia/sketch/thermal/cartoon/night_vision/edge_glow)
 *   - CLAHE
 *   - Harris corners
 *   - Lucas-Kanade optical flow
 *   - Background subtractor
 *   - HOG descriptor
 *   - Template matching
 *   - Contour detection
 *   - Histogram + dominant colors
 *   - V4L2 Camera (Linux)
 *   - SIMD capabilities query
 *   - Memory pool stats
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

// liteCV6 headers
#include "liteCV/core/lcvdef.hpp"
#include "liteCV/core/lcvtypes.hpp"
#include "liteCV/core/matrix.hpp"
#include "liteCV/core/core.hpp"
#include "liteCV/imgcodecs/imgcodecs.hpp"
#include "liteCV/imgproc/imgproc.hpp"
#include "liteCV/imgproc/color.hpp"
#include "liteCV/imgproc/filter.hpp"
#include "liteCV/imgproc/transform.hpp"

// Advanced headers
#include "lcv_simd.hpp"
#include "lcv_mempool.hpp"
#include "lcv_advanced.hpp"
#include "lcv_morph.hpp"
#include "lcv_camera.hpp"

#ifdef SSD
#undef SSD
#endif
#ifdef NCC
#undef NCC
#endif

namespace py = pybind11;

// ── Helpers: numpy ↔ lcv::Mat ─────────────────────────────────────────────
static lcv::Mat numpy_to_mat(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    if (info.ndim == 2) {
        int h = info.shape[0], w = info.shape[1];
        lcv::Mat m(w, h, 1, "uint8");
        std::memcpy(m.data, info.ptr, h * w);
        return m;
    } else if (info.ndim == 3) {
        int h = info.shape[0], w = info.shape[1], ch = info.shape[2];
        lcv::Mat m(w, h, ch, "uint8");
        std::memcpy(m.data, info.ptr, h * w * ch);
        return m;
    }
    throw std::runtime_error("Array must be 2D (gray) or 3D (color)");
}

static py::array_t<uint8_t> mat_to_numpy(const lcv::Mat& m) {
    if (m.channels() == 1)
        return py::array_t<uint8_t>({m.rows, m.cols}, m.data);
    return py::array_t<uint8_t>({m.rows, m.cols, m.channels()}, m.data);
}

// ── Resize (work around liteCV6 assert bug) ─────────────────────────────
static lcv::Mat safe_resize(const lcv::Mat& src, int new_w, int new_h) {
    lcv::Mat dst(new_w, new_h, src.channels(), "uint8");
    float sx = (float)(src.cols)  / new_w;
    float sy = (float)(src.rows) / new_h;
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int sy2 = std::min((int)(y * sy), src.rows - 1);
            int sx2 = std::min((int)(x * sx), src.cols - 1);
            for (int c = 0; c < src.channels(); c++)
                dst.data[(y * new_w + x) * dst.channels() + c] =
                    src.data[(sy2 * src.cols + sx2) * src.channels() + c];
        }
    }
    return dst;
}

// ── Color-space helpers (RGB-centric API, liteCV6 uses BGR internally) ───
// liteCV6 color codes
static const int LCV_RGB2GRAY = 5;  // actually RGB2GRAY in liteCV6 color.hpp
static const int LCV_BGR2GRAY = 6;

// ── CLAHE effect (single or multi-channel) ───────────────────────────────
static py::array_t<uint8_t> py_clahe(
    py::array_t<uint8_t, py::array::c_style> arr,
    int tile_w, int tile_h, float clip_limit)
{
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;

    std::vector<uint8_t> dst(h * w * ch);
    if (ch == 1) {
        lcv_adv::clahe(src, dst.data(), w, h, tile_w, tile_h, clip_limit);
    } else {
        // Apply per channel
        std::vector<uint8_t> ch_src(h * w), ch_dst(h * w);
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < h * w; i++) ch_src[i] = src[i * ch + c];
            lcv_adv::clahe(ch_src.data(), ch_dst.data(), w, h, tile_w, tile_h, clip_limit);
            for (int i = 0; i < h * w; i++) dst[i * ch + c] = ch_dst[i];
        }
    }

    if (ch == 1) return py::array_t<uint8_t>({h, w}, dst.data());
    return py::array_t<uint8_t>({h, w, ch}, dst.data());
}

// ── Bilateral filter ─────────────────────────────────────────────────────
static py::array_t<uint8_t> py_bilateral(
    py::array_t<uint8_t, py::array::c_style> arr,
    int d, float sigma_color, float sigma_space)
{
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;
    std::vector<uint8_t> dst(h * w * ch);

    if (ch == 1) {
        lcv_adv::bilateral_filter(src, dst.data(), w, h, d, sigma_color, sigma_space);
    } else {
        std::vector<uint8_t> ch_src(h*w), ch_dst(h*w);
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < h*w; i++) ch_src[i] = src[i*ch+c];
            lcv_adv::bilateral_filter(ch_src.data(), ch_dst.data(), w, h, d, sigma_color, sigma_space);
            for (int i = 0; i < h*w; i++) dst[i*ch+c] = ch_dst[i];
        }
    }
    if (ch == 1) return py::array_t<uint8_t>({h, w}, dst.data());
    return py::array_t<uint8_t>({h, w, ch}, dst.data());
}

// ── Morphological operations ──────────────────────────────────────────────
static py::array_t<uint8_t> py_morph(
    py::array_t<uint8_t, py::array::c_style> arr,
    const std::string& op, int ksize, const std::string& shape_str)
{
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    const uint8_t* src = (const uint8_t*)info.ptr;
    std::vector<uint8_t> dst(h * w), tmp1(h * w), tmp2(h * w);

    lcv_morph::MorphShape shape = lcv_morph::MorphShape::RECT;
    if (shape_str == "ellipse") shape = lcv_morph::MorphShape::ELLIPSE;
    else if (shape_str == "cross") shape = lcv_morph::MorphShape::CROSS;

    if      (op == "erode")    lcv_morph::erode(src, dst.data(), w, h, ksize, shape);
    else if (op == "dilate")   lcv_morph::dilate(src, dst.data(), w, h, ksize, shape);
    else if (op == "open")     lcv_morph::morph_open(src, dst.data(), tmp1.data(), w, h, ksize, shape);
    else if (op == "close")    lcv_morph::morph_close(src, dst.data(), tmp1.data(), w, h, ksize, shape);
    else if (op == "gradient") lcv_morph::morph_gradient(src, dst.data(), tmp1.data(), tmp2.data(), w, h, ksize, shape);
    else if (op == "tophat")   lcv_morph::tophat(src, dst.data(), tmp1.data(), tmp2.data(), w, h, ksize, shape);
    else if (op == "blackhat") lcv_morph::blackhat(src, dst.data(), tmp1.data(), tmp2.data(), w, h, ksize, shape);
    else throw std::runtime_error("Unknown morphological op: " + op);

    return py::array_t<uint8_t>({h, w}, dst.data());
}

// ── Harris corners ────────────────────────────────────────────────────────
static py::list py_harris(
    py::array_t<uint8_t, py::array::c_style> arr,
    int block_size, int ksize, float k, float threshold, int max_corners)
{
    auto info = arr.request();
    if (info.ndim != 2) throw std::runtime_error("Harris expects grayscale (2D) array");
    int h = info.shape[0], w = info.shape[1];
    auto corners = lcv_adv::harris_corners(
        (const uint8_t*)info.ptr, w, h,
        block_size, ksize, k, threshold, max_corners);
    py::list result;
    for (auto& c : corners)
        result.append(py::make_tuple(c.x, c.y, c.score));
    return result;
}

// ── Lucas-Kanade optical flow ─────────────────────────────────────────────
static py::list py_lk_flow(
    py::array_t<uint8_t, py::array::c_style> prev_arr,
    py::array_t<uint8_t, py::array::c_style> next_arr,
    py::list pts_list, int win_size, int max_iter, float eps)
{
    auto pi = prev_arr.request(), ni = next_arr.request();
    if (pi.ndim != 2 || ni.ndim != 2) throw std::runtime_error("LK flow expects grayscale arrays");
    int h = pi.shape[0], w = pi.shape[1];
    std::vector<std::pair<float,float>> pts;
    for (auto& p : pts_list) {
        auto t = py::cast<py::tuple>(p);
        pts.push_back({py::cast<float>(t[0]), py::cast<float>(t[1])});
    }
    auto flows = lcv_adv::lk_optical_flow(
        (const uint8_t*)pi.ptr, (const uint8_t*)ni.ptr,
        pts, w, h, win_size, max_iter, eps);
    py::list result;
    for (auto& f : flows)
        result.append(py::make_tuple(f.x, f.y, f.u, f.v, f.valid));
    return result;
}

// ── HOG descriptor ───────────────────────────────────────────────────────
static py::list py_hog(
    py::array_t<uint8_t, py::array::c_style> arr,
    int cell_size, int block_size, int num_bins)
{
    auto info = arr.request();
    if (info.ndim != 2) throw std::runtime_error("HOG expects grayscale (2D) array");
    int h = info.shape[0], w = info.shape[1];
    lcv_adv::HOGParams p{cell_size, block_size, num_bins};
    auto desc = lcv_adv::compute_hog((const uint8_t*)info.ptr, w, h, p);
    py::list result;
    for (float v : desc) result.append(v);
    return result;
}

// ── Template matching ─────────────────────────────────────────────────────
static py::tuple py_template_match(
    py::array_t<uint8_t, py::array::c_style> img_arr,
    py::array_t<uint8_t, py::array::c_style> tmpl_arr,
    const std::string& method)
{
    auto ii = img_arr.request(), ti = tmpl_arr.request();
    if (ii.ndim != 2 || ti.ndim != 2) throw std::runtime_error("template_match expects grayscale arrays");
    auto m = (method == "ssd") ? lcv_adv::MatchMethod::SSD : lcv_adv::MatchMethod::NCC;
    auto res = lcv_adv::template_match(
        (const uint8_t*)ii.ptr, ii.shape[1], ii.shape[0],
        (const uint8_t*)ti.ptr, ti.shape[1], ti.shape[0], m);
    return py::make_tuple(res.x, res.y, res.score);
}

// ── Contour detection ──────────────────────────────────────────────────────
static py::list py_find_contours(
    py::array_t<uint8_t, py::array::c_style> arr, int min_len)
{
    auto info = arr.request();
    if (info.ndim != 2) throw std::runtime_error("find_contours expects 2D binary mask");
    int h = info.shape[0], w = info.shape[1];
    auto contours = lcv_adv::find_contours((const uint8_t*)info.ptr, w, h, min_len);
    py::list result;
    for (auto& c : contours) {
        auto [x, y, cw, ch2] = lcv_adv::contour_bbox(c);
        float area = lcv_adv::contour_area(c);
        py::list pts;
        for (auto& [px, py2] : c) pts.append(py::make_tuple(px, py2));
        py::dict d;
        d["points"] = pts;
        d["bbox"] = py::make_tuple(x, y, cw, ch2);
        d["area"] = area;
        d["length"] = (int)c.size();
        result.append(d);
    }
    return result;
}

// ── Histogram + equalization ──────────────────────────────────────────────
static py::list py_histogram(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    int n = 1;
    for (auto s : info.shape) n *= s;
    auto hist = lcv_adv::compute_histogram((const uint8_t*)info.ptr, n);
    py::list result;
    for (int v : hist) result.append(v);
    return result;
}

static py::array_t<uint8_t> py_equalize(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    if (info.ndim != 2) throw std::runtime_error("equalize expects grayscale 2D array");
    int h = info.shape[0], w = info.shape[1];
    std::vector<uint8_t> dst(h * w);
    lcv_adv::equalize_histogram((const uint8_t*)info.ptr, dst.data(), h*w);
    return py::array_t<uint8_t>({h, w}, dst.data());
}

// ── Dominant colors ───────────────────────────────────────────────────────
static py::list py_dominant_colors(
    py::array_t<uint8_t, py::array::c_style> arr, int num_colors)
{
    auto info = arr.request();
    if (info.ndim != 3 || info.shape[2] != 3)
        throw std::runtime_error("dominant_colors expects RGB (H,W,3) array");
    int num_pixels = info.shape[0] * info.shape[1];
    auto colors = lcv_adv::dominant_colors((const uint8_t*)info.ptr, num_pixels, num_colors);
    py::list result;
    for (auto& c : colors) {
        py::dict d;
        d["r"] = c.r;
        d["g"] = c.g;
        d["b"] = c.b;
        d["fraction"] = c.fraction;
        char buf[8];
        snprintf(buf, sizeof(buf), "%02X%02X%02X", c.r, c.g, c.b);
        d["hex"] = "#" + std::string(buf);
        result.append(d);
    }
    return result;
}

// ── Brightness / contrast via SIMD ───────────────────────────────────────
static py::array_t<uint8_t> py_brightness(
    py::array_t<uint8_t, py::array::c_style> arr, int delta)
{
    auto info = arr.request();
    int n = 1; for (auto s : info.shape) n *= s;
    std::vector<uint8_t> dst(n);
    lcv_simd::add_saturate((const uint8_t*)info.ptr, dst.data(), n, delta);
    if (info.ndim == 2) return py::array_t<uint8_t>({info.shape[0], info.shape[1]}, dst.data());
    return py::array_t<uint8_t>({info.shape[0], info.shape[1], info.shape[2]}, dst.data());
}

static py::array_t<uint8_t> py_contrast(
    py::array_t<uint8_t, py::array::c_style> arr, float scale)
{
    auto info = arr.request();
    int n = 1; for (auto s : info.shape) n *= s;
    std::vector<uint8_t> dst(n);
    // contrast: (pixel - 128) * scale + 128
    const uint8_t* src = (const uint8_t*)info.ptr;
    for (int i = 0; i < n; i++) {
        float v = (src[i] - 128.f) * scale + 128.f;
        dst[i] = lcv_simd::clamp_u8((int)v);
    }
    if (info.ndim == 2) return py::array_t<uint8_t>({info.shape[0], info.shape[1]}, dst.data());
    return py::array_t<uint8_t>({info.shape[0], info.shape[1], info.shape[2]}, dst.data());
}

// ── Grayscale (SIMD accelerated) ─────────────────────────────────────────
static py::array_t<uint8_t> py_to_gray(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    if (info.ndim == 2) return arr; // already gray
    if (info.ndim != 3 || info.shape[2] != 3)
        throw std::runtime_error("to_gray expects RGB (H,W,3) array");
    int h = info.shape[0], w = info.shape[1];
    std::vector<uint8_t> dst(h * w);
    lcv_simd::rgb_to_gray((const uint8_t*)info.ptr, dst.data(), h * w);
    return py::array_t<uint8_t>({h, w}, dst.data());
}

// ── Absolute difference ───────────────────────────────────────────────────
static py::array_t<uint8_t> py_absdiff(
    py::array_t<uint8_t, py::array::c_style> a,
    py::array_t<uint8_t, py::array::c_style> b)
{
    auto ai = a.request(), bi = b.request();
    if (ai.size != bi.size) throw std::runtime_error("absdiff: arrays must have same size");
    int n = ai.size;
    std::vector<uint8_t> dst(n);
    lcv_simd::absdiff((const uint8_t*)ai.ptr, (const uint8_t*)bi.ptr, dst.data(), n);
    if (ai.ndim == 2) return py::array_t<uint8_t>({ai.shape[0], ai.shape[1]}, dst.data());
    return py::array_t<uint8_t>({ai.shape[0], ai.shape[1], ai.shape[2]}, dst.data());
}

// ── Unsharp mask ──────────────────────────────────────────────────────────
static py::array_t<uint8_t> py_unsharp(
    py::array_t<uint8_t, py::array::c_style> arr, float amount, int radius)
{
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    std::vector<uint8_t> dst(h * w * ch);
    lcv_adv::unsharp_mask((const uint8_t*)info.ptr, dst.data(), w, h, ch, amount, radius);
    if (ch == 1) return py::array_t<uint8_t>({h, w}, dst.data());
    return py::array_t<uint8_t>({h, w, ch}, dst.data());
}

// ── imread / imwrite / resize / rotate / flip ─────────────────────────────
static py::array_t<uint8_t> py_imread(const std::string& path, bool grayscale) {
    lcv::Mat m = lcv::imread(path, grayscale ? 0 : 1);
    if (m.data == nullptr) throw std::runtime_error("imread failed: " + path);
    // liteCV6 returns BGR; convert to RGB
    if (!grayscale && m.channels() == 3) {
        for (int i = 0; i < m.rows * m.cols; i++)
            std::swap(m.data[i*3+0], m.data[i*3+2]);
    }
    return mat_to_numpy(m);
}

static void py_imwrite(const std::string& path, py::array_t<uint8_t, py::array::c_style> arr) {
    lcv::Mat m = numpy_to_mat(arr);
    // Convert RGB → BGR for liteCV6
    if (m.channels() == 3) {
        for (int i = 0; i < m.rows * m.cols; i++)
            std::swap(m.data[i*3+0], m.data[i*3+2]);
    }
    if (!lcv::imwrite(path, m))
        throw std::runtime_error("imwrite failed: " + path);
}

static py::array_t<uint8_t> py_imdecode(py::bytes data) {
    std::string s = data;
    std::vector<uchar> buf((uint8_t*)s.data(), (uint8_t*)s.data() + s.size());
    lcv::Mat m = lcv::imdecode(buf);
    if (m.data == nullptr) throw std::runtime_error("imdecode failed");
    if (m.channels() == 3)
        for (int i = 0; i < m.rows * m.cols; i++) std::swap(m.data[i*3+0], m.data[i*3+2]);
    return mat_to_numpy(m);
}

static py::bytes py_imencode(py::array_t<uint8_t, py::array::c_style> arr, const std::string& ext) {
    lcv::Mat m = numpy_to_mat(arr);
    if (m.channels() == 3)
        for (int i = 0; i < m.rows * m.cols; i++) std::swap(m.data[i*3+0], m.data[i*3+2]);
    std::vector<uchar> encoded;
    if (!lcv::imencode(ext, m, encoded))
        throw std::runtime_error("imencode failed");
    return py::bytes((char*)encoded.data(), encoded.size());
}

static py::array_t<uint8_t> py_resize(
    py::array_t<uint8_t, py::array::c_style> arr, int new_w, int new_h)
{
    lcv::Mat m = numpy_to_mat(arr);
    lcv::Mat r = safe_resize(m, new_w, new_h);
    return mat_to_numpy(r);
}

static py::array_t<uint8_t> py_rotate(
    py::array_t<uint8_t, py::array::c_style> arr, int angle_code)
{
    // angle_code: 0=90CW, 1=180, 2=90CCW
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;

    if (angle_code == 1) { // 180°
        std::vector<uint8_t> dst(h * w * ch);
        for (int i = 0; i < h * w * ch; i++)
            dst[i] = src[(h*w*ch - 1 - i/ch)*ch + i%ch];
        if (ch == 1) return py::array_t<uint8_t>({h, w}, dst.data());
        return py::array_t<uint8_t>({h, w, ch}, dst.data());
    } else if (angle_code == 0) { // 90° CW
        std::vector<uint8_t> dst(h * w * ch);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < ch; c++)
                    dst[(x * h + (h - 1 - y)) * ch + c] = src[(y * w + x) * ch + c];
        if (ch == 1) return py::array_t<uint8_t>({w, h}, dst.data());
        return py::array_t<uint8_t>({w, h, ch}, dst.data());
    } else { // 90° CCW
        std::vector<uint8_t> dst(h * w * ch);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < ch; c++)
                    dst[((w - 1 - x) * h + y) * ch + c] = src[(y * w + x) * ch + c];
        if (ch == 1) return py::array_t<uint8_t>({w, h}, dst.data());
        return py::array_t<uint8_t>({w, h, ch}, dst.data());
    }
}

static py::array_t<uint8_t> py_flip(
    py::array_t<uint8_t, py::array::c_style> arr, int flip_code)
{
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;
    std::vector<uint8_t> dst(h * w * ch);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int ny = (flip_code == 0 || flip_code == -1) ? (h - 1 - y) : y;
            int nx = (flip_code == 1 || flip_code == -1) ? (w - 1 - x) : x;
            for (int c = 0; c < ch; c++)
                dst[(ny * w + nx) * ch + c] = src[(y * w + x) * ch + c];
        }
    }
    if (ch == 1) return py::array_t<uint8_t>({h, w}, dst.data());
    return py::array_t<uint8_t>({h, w, ch}, dst.data());
}

// ── Effects (C++-only, fast) ──────────────────────────────────────────────
static py::array_t<uint8_t> py_sepia(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    if (info.ndim != 3 || info.shape[2] != 3)
        throw std::runtime_error("sepia expects RGB (H,W,3)");
    int n = info.shape[0] * info.shape[1];
    const uint8_t* src = (const uint8_t*)info.ptr;
    std::vector<uint8_t> dst(n * 3);
    for (int i = 0; i < n; i++) {
        float r = src[i*3], g = src[i*3+1], b = src[i*3+2];
        dst[i*3+0] = lcv_simd::clamp_u8((int)(r*0.393f + g*0.769f + b*0.189f));
        dst[i*3+1] = lcv_simd::clamp_u8((int)(r*0.349f + g*0.686f + b*0.168f));
        dst[i*3+2] = lcv_simd::clamp_u8((int)(r*0.272f + g*0.534f + b*0.131f));
    }
    return py::array_t<uint8_t>({info.shape[0], info.shape[1], (pybind11::ssize_t)3}, dst.data());
}

static py::array_t<uint8_t> py_thermal(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    if (info.ndim != 3 || info.shape[2] != 3)
        throw std::runtime_error("thermal expects RGB (H,W,3)");
    int n = info.shape[0] * info.shape[1];
    const uint8_t* src = (const uint8_t*)info.ptr;
    std::vector<uint8_t> gray(n);
    lcv_simd::rgb_to_gray(src, gray.data(), n);
    std::vector<uint8_t> dst(n * 3);
    // Iron colormap approximation
    for (int i = 0; i < n; i++) {
        float v = gray[i] / 255.f;
        dst[i*3+0] = lcv_simd::clamp_u8((int)(255 * std::min(1.f, 3.f * v)));
        dst[i*3+1] = lcv_simd::clamp_u8((int)(255 * std::max(0.f, std::min(1.f, 3.f*v - 1.f))));
        dst[i*3+2] = lcv_simd::clamp_u8((int)(255 * std::max(0.f, std::min(1.f, 3.f*v - 2.f))));
    }
    return py::array_t<uint8_t>({info.shape[0], info.shape[1], (pybind11::ssize_t)3}, dst.data());
}

static py::array_t<uint8_t> py_night_vision(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    int n = info.shape[0] * info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;
    std::vector<uint8_t> gray(n);
    if (ch >= 3) lcv_simd::rgb_to_gray(src, gray.data(), n);
    else std::copy(src, src+n, gray.data());
    std::vector<uint8_t> dst(n * 3);
    // CLAHE enhance first
    std::vector<uint8_t> enhanced(n);
    lcv_adv::clahe(gray.data(), enhanced.data(),
                   info.shape[1], info.shape[0], 8, 8, 3.0f);
    for (int i = 0; i < n; i++) {
        dst[i*3+0] = 0;
        dst[i*3+1] = enhanced[i]; // Green channel only (night vision look)
        dst[i*3+2] = 0;
    }
    return py::array_t<uint8_t>({info.shape[0], info.shape[1], (pybind11::ssize_t)3}, dst.data());
}

static py::array_t<uint8_t> py_sketch(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;
    int n = h * w;
    std::vector<uint8_t> gray(n);
    if (ch >= 3) lcv_simd::rgb_to_gray(src, gray.data(), n);
    else std::copy(src, src+n, gray.data());

    // Sobel edges
    std::vector<uint8_t> edges(n, 0);
    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            int gx = -gray[(y-1)*w+x-1] + gray[(y-1)*w+x+1]
                     - 2*gray[y*w+x-1]  + 2*gray[y*w+x+1]
                     - gray[(y+1)*w+x-1] + gray[(y+1)*w+x+1];
            int gy = -gray[(y-1)*w+x-1] - 2*gray[(y-1)*w+x] - gray[(y-1)*w+x+1]
                     + gray[(y+1)*w+x-1] + 2*gray[(y+1)*w+x] + gray[(y+1)*w+x+1];
            int mag = (int)std::sqrt((float)(gx*gx + gy*gy));
            edges[y*w+x] = lcv_simd::clamp_u8(mag);
        }
    }
    // Invert for sketch look
    std::vector<uint8_t> dst(n * 3);
    for (int i = 0; i < n; i++) {
        uint8_t v = 255 - edges[i];
        dst[i*3+0] = dst[i*3+1] = dst[i*3+2] = v;
    }
    return py::array_t<uint8_t>({h, w, 3}, dst.data());
}

static py::array_t<uint8_t> py_cartoon(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    if (info.ndim != 3 || info.shape[2] != 3)
        throw std::runtime_error("cartoon expects RGB (H,W,3)");
    const uint8_t* src = (const uint8_t*)info.ptr;
    int n = h * w;

    // Gray + edges
    std::vector<uint8_t> gray(n);
    lcv_simd::rgb_to_gray(src, gray.data(), n);

    // Bilateral smooth the color image (3 iterations, lighter params for speed)
    std::vector<uint8_t> smooth(n * 3);
    for (int c = 0; c < 3; c++) {
        std::vector<uint8_t> ch_src(n), ch_dst(n);
        for (int i = 0; i < n; i++) ch_src[i] = src[i*3+c];
        lcv_adv::bilateral_filter(ch_src.data(), ch_dst.data(), w, h, 7, 50.f, 50.f);
        for (int i = 0; i < n; i++) smooth[i*3+c] = ch_dst[i];
    }

    // Quantize color (posterize to 4 levels)
    for (int i = 0; i < n*3; i++)
        smooth[i] = (uint8_t)(((smooth[i] / 64) * 64) + 32);

    // Edge mask
    std::vector<uint8_t> edges(n, 0);
    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            int gx = (int)gray[(y-1)*w+x+1] - gray[(y-1)*w+x-1]
                   + 2*(gray[y*w+x+1] - gray[y*w+x-1])
                   + gray[(y+1)*w+x+1] - gray[(y+1)*w+x-1];
            int gy = (int)gray[(y+1)*w+x-1] - gray[(y-1)*w+x-1]
                   + 2*(gray[(y+1)*w+x] - gray[(y-1)*w+x])
                   + gray[(y+1)*w+x+1] - gray[(y-1)*w+x+1];
            edges[y*w+x] = (std::abs(gx) + std::abs(gy) > 60) ? 0 : 255;
        }
    }

    // Combine
    std::vector<uint8_t> dst(n * 3);
    for (int i = 0; i < n; i++) {
        float mask = edges[i] / 255.f;
        dst[i*3+0] = (uint8_t)(smooth[i*3+0] * mask);
        dst[i*3+1] = (uint8_t)(smooth[i*3+1] * mask);
        dst[i*3+2] = (uint8_t)(smooth[i*3+2] * mask);
    }
    return py::array_t<uint8_t>({h, w, 3}, dst.data());
}

static py::array_t<uint8_t> py_edge_glow(
    py::array_t<uint8_t, py::array::c_style> arr, uint8_t r, uint8_t g, uint8_t b)
{
    auto info = arr.request();
    int h = info.shape[0], w = info.shape[1];
    int ch = (info.ndim == 3) ? info.shape[2] : 1;
    const uint8_t* src = (const uint8_t*)info.ptr;
    int n = h * w;
    std::vector<uint8_t> gray(n);
    if (ch >= 3) lcv_simd::rgb_to_gray(src, gray.data(), n);
    else std::copy(src, src+n, gray.data());

    std::vector<uint8_t> dst(n * 3, 0);
    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {
            int gx = (int)gray[y*w+x+1] - gray[y*w+x-1];
            int gy2 = (int)gray[(y+1)*w+x] - gray[(y-1)*w+x];
            float mag = std::sqrt((float)(gx*gx + gy2*gy2)) / 362.f; // normalize
            dst[(y*w+x)*3+0] = lcv_simd::clamp_u8((int)(r * mag));
            dst[(y*w+x)*3+1] = lcv_simd::clamp_u8((int)(g * mag));
            dst[(y*w+x)*3+2] = lcv_simd::clamp_u8((int)(b * mag));
        }
    }
    return py::array_t<uint8_t>({h, w, 3}, dst.data());
}

// ── V4L2 Camera class binding ─────────────────────────────────────────────
class PyCamera {
    lcv_cam::Camera cam;
public:
    PyCamera(const std::string& dev, int w, int h, bool prefer_mjpeg)
        : cam(dev, w, h, prefer_mjpeg) {}

    py::array_t<uint8_t> capture() {
        auto rgb = cam.capture_frame();
        int w = cam.get_width(), h = cam.get_height();
        return py::array_t<uint8_t>({h, w, 3}, rgb.data());
    }

    int width()  const { return cam.get_width(); }
    int height() const { return cam.get_height(); }
    std::string fmt() const { return cam.format_name(); }
};

// ── Background subtractor binding ─────────────────────────────────────────
class PyBGSub {
    lcv_adv::BackgroundSubtractor sub;
public:
    PyBGSub(float lr, float thr) : sub(lr, thr) {}

    py::array_t<uint8_t> apply(py::array_t<uint8_t, py::array::c_style> arr) {
        auto info = arr.request();
        if (info.ndim != 2) throw std::runtime_error("BGSub.apply expects grayscale 2D array");
        int h = info.shape[0], w = info.shape[1];
        std::vector<uint8_t> mask(h * w);
        sub.apply((const uint8_t*)info.ptr, mask.data(), w, h);
        return py::array_t<uint8_t>({h, w}, mask.data());
    }

    void reset() { sub.reset(); }
    void set_learning_rate(float lr) { sub.learning_rate = lr; }
    void set_threshold(float thr)    { sub.threshold = thr; }
};

// ═══════════════════════════════════════════════════════════════════════════
//  PYBIND11 MODULE DEFINITION
// ═══════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(_lcv_ext, m) {
    m.doc() = "litecv-advanced C++ extension (pybind11) — edge-optimized vision";

    // ── System info ─────────────────────────────────────────────────────
    m.def("simd_backend", []() { return std::string(lcv_simd::simd_name()); },
          "Return the active SIMD backend name (NEON / SSE2 / AVX2 / scalar)");

    m.def("pool_stats", []() -> py::dict {
        py::dict d;
        d["frame_used_mb"] = lcv_pool::g_frame_pool.used / (1024.f*1024.f);
        d["frame_total_mb"] = (float)lcv_pool::FRAME_POOL_SIZE / (1024.f*1024.f);
        d["scratch_used_mb"] = lcv_pool::g_scratch_pool.used / (1024.f*1024.f);
        return d;
    }, "Memory pool usage statistics");

    m.def("pool_reset", []() {
        lcv_pool::frame_reset();
        lcv_pool::scratch_reset();
    }, "Reset memory pools (call after each frame in tight loops)");

    // ── I/O ─────────────────────────────────────────────────────────────
    m.def("imread",   &py_imread,   py::arg("path"), py::arg("grayscale")=false);
    m.def("imwrite",  &py_imwrite,  py::arg("path"), py::arg("array"));
    m.def("imdecode", &py_imdecode, py::arg("data"));
    m.def("imencode", &py_imencode, py::arg("array"), py::arg("ext")=".jpg");

    // ── Geometry ─────────────────────────────────────────────────────────
    m.def("resize", &py_resize, py::arg("array"), py::arg("width"), py::arg("height"));
    m.def("rotate", &py_rotate, py::arg("array"), py::arg("angle_code"),
          "angle_code: 0=90CW, 1=180, 2=90CCW");
    m.def("flip",   &py_flip,   py::arg("array"), py::arg("flip_code"),
          "flip_code: 0=vertical, 1=horizontal, -1=both");

    // ── Color / channel ───────────────────────────────────────────────────
    m.def("to_gray", &py_to_gray, py::arg("array"), "RGB (H,W,3) → grayscale (H,W)");

    // ── Pixel-level ops (SIMD) ────────────────────────────────────────────
    m.def("brightness", &py_brightness, py::arg("array"), py::arg("delta"),
          "Add delta to all pixels (SIMD-accelerated, clamped 0-255)");
    m.def("contrast",   &py_contrast,   py::arg("array"), py::arg("scale"),
          "Scale contrast around mid-point 128");
    m.def("absdiff",    &py_absdiff,    py::arg("a"), py::arg("b"),
          "Absolute difference between two arrays (SIMD-accelerated)");

    // ── Filters ───────────────────────────────────────────────────────────
    m.def("bilateral",  &py_bilateral,  py::arg("array"),
          py::arg("d")=9, py::arg("sigma_color")=75.f, py::arg("sigma_space")=75.f,
          "Edge-preserving bilateral filter");
    m.def("clahe",      &py_clahe,      py::arg("array"),
          py::arg("tile_w")=8, py::arg("tile_h")=8, py::arg("clip_limit")=2.0f,
          "CLAHE: Contrast Limited Adaptive Histogram Equalization");
    m.def("unsharp",    &py_unsharp,    py::arg("array"),
          py::arg("amount")=1.5f, py::arg("radius")=3,
          "Unsharp mask sharpening");

    // ── Morphological ops (grayscale) ─────────────────────────────────────
    m.def("morph", &py_morph,
          py::arg("array"), py::arg("op"), py::arg("ksize")=3,
          py::arg("shape")="rect",
          "Morphological op: op = erode|dilate|open|close|gradient|tophat|blackhat");

    // ── Histogram ────────────────────────────────────────────────────────
    m.def("histogram", &py_histogram, py::arg("array"),
          "Compute 256-bin histogram of a grayscale array");
    m.def("equalize",  &py_equalize,  py::arg("array"),
          "Global histogram equalization (grayscale)");

    // ── Harris corners ────────────────────────────────────────────────────
    m.def("harris_corners", &py_harris,
          py::arg("gray"), py::arg("block_size")=3, py::arg("ksize")=3,
          py::arg("k")=0.04f, py::arg("threshold")=0.01f, py::arg("max_corners")=500,
          "Harris corner detection → list of (x, y, score)");

    // ── Optical flow ──────────────────────────────────────────────────────
    m.def("lk_optical_flow", &py_lk_flow,
          py::arg("prev"), py::arg("next"), py::arg("points"),
          py::arg("win_size")=15, py::arg("max_iter")=30, py::arg("eps")=0.01f,
          "Lucas-Kanade sparse optical flow → list of (x, y, u, v, valid)");

    // ── HOG descriptor ────────────────────────────────────────────────────
    m.def("hog_descriptor", &py_hog,
          py::arg("gray"), py::arg("cell_size")=8,
          py::arg("block_size")=2, py::arg("num_bins")=9,
          "Compute HOG feature descriptor → list of floats");

    // ── Template matching ─────────────────────────────────────────────────
    m.def("template_match", &py_template_match,
          py::arg("image"), py::arg("template"), py::arg("method")="ncc",
          "Template matching → (x, y, score). method: 'ssd' or 'ncc'");

    // ── Contour detection ─────────────────────────────────────────────────
    m.def("find_contours", &py_find_contours,
          py::arg("mask"), py::arg("min_len")=10,
          "Find contours in binary mask → list of dicts with points/bbox/area");

    // ── Dominant colors ───────────────────────────────────────────────────
    m.def("dominant_colors", &py_dominant_colors,
          py::arg("rgb_array"), py::arg("num_colors")=5,
          "Find dominant colors → list of dicts with r/g/b/fraction/hex");

    // ── Effects ───────────────────────────────────────────────────────────
    m.def("effect_sepia",        &py_sepia,        py::arg("array"));
    m.def("effect_thermal",      &py_thermal,      py::arg("array"));
    m.def("effect_night_vision", &py_night_vision, py::arg("array"));
    m.def("effect_sketch",       &py_sketch,       py::arg("array"));
    m.def("effect_cartoon",      &py_cartoon,      py::arg("array"));
    m.def("effect_edge_glow",    &py_edge_glow,
          py::arg("array"), py::arg("r")=0, py::arg("g")=255, py::arg("b")=120);

    // ── V4L2 Camera ───────────────────────────────────────────────────────
    py::class_<PyCamera>(m, "Camera")
        .def(py::init<const std::string&, int, int, bool>(),
             py::arg("device"), py::arg("width"), py::arg("height"),
             py::arg("prefer_mjpeg")=true)
        .def("capture", &PyCamera::capture, "Capture one frame as numpy RGB array")
        .def_property_readonly("width",  &PyCamera::width)
        .def_property_readonly("height", &PyCamera::height)
        .def_property_readonly("format", &PyCamera::fmt);

    // ── Background subtractor ─────────────────────────────────────────────
    py::class_<PyBGSub>(m, "BackgroundSubtractor")
        .def(py::init<float, float>(),
             py::arg("learning_rate")=0.05f, py::arg("threshold")=25.f)
        .def("apply",              &PyBGSub::apply,
             py::arg("gray_array"), "Apply to grayscale frame → foreground mask")
        .def("reset",              &PyBGSub::reset)
        .def("set_learning_rate",  &PyBGSub::set_learning_rate, py::arg("lr"))
        .def("set_threshold",      &PyBGSub::set_threshold,     py::arg("thr"));
}
