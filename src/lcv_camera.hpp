/*
 * lcv_camera.hpp — V4L2 camera with mmap zero-copy DMA buffers
 *
 * Linux-only (Raspberry Pi, Jetson, any V4L2 device).
 * Zero-copy: frames are memory-mapped directly from kernel buffers.
 * Supports MJPEG (default) and YUYV formats.
 * Throws std::runtime_error on non-Linux or if device fails.
 */
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <functional>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <cerrno>
#include <cstring>

// STB decode for MJPEG
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "liteCV/imgcodecs/stb/stb_image.h"

namespace lcv_cam {

struct MmapBuffer {
    void*  start  = nullptr;
    size_t length = 0;
};

class Camera {
public:
    int            fd       = -1;
    int            width    = 0;
    int            height   = 0;
    int            fmt      = 0; // V4L2_PIX_FMT_MJPEG or YUYV
    int            num_bufs = 4;
    std::vector<MmapBuffer> buffers;

    // Constructor: opens device and negotiates format
    Camera(const std::string& device, int w, int h, bool prefer_mjpeg = true)
        : width(w), height(h)
    {
        fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0)
            throw std::runtime_error("Cannot open camera: " + device + " (" + strerror(errno) + ")");

        // Query capabilities
        v4l2_capability cap{};
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0)
            throw std::runtime_error("VIDIOC_QUERYCAP failed");
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
            throw std::runtime_error("Device is not a video capture device");

        // Try to set format
        if (prefer_mjpeg) {
            if (!try_set_format(V4L2_PIX_FMT_MJPEG)) {
                if (!try_set_format(V4L2_PIX_FMT_YUYV))
                    throw std::runtime_error("Neither MJPEG nor YUYV format supported");
            }
        } else {
            if (!try_set_format(V4L2_PIX_FMT_YUYV)) {
                if (!try_set_format(V4L2_PIX_FMT_MJPEG))
                    throw std::runtime_error("Neither YUYV nor MJPEG format supported");
            }
        }

        init_mmap();
        start_streaming();
    }

    ~Camera() {
        if (fd >= 0) {
            stop_streaming();
            for (auto& buf : buffers)
                if (buf.start) munmap(buf.start, buf.length);
            close(fd);
        }
    }

    // Capture one frame. Returns RGB bytes (width*height*3).
    // Uses zero-copy mmap — no extra allocation inside kernel.
    std::vector<uint8_t> capture_frame() {
        // Dequeue
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        // Wait for frame (blocking with select)
        fd_set fds; FD_ZERO(&fds); FD_SET(fd, &fds);
        timeval tv{2, 0}; // 2 second timeout
        int r = select(fd + 1, &fds, nullptr, nullptr, &tv);
        if (r <= 0) throw std::runtime_error("Camera frame timeout or error");

        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0)
            throw std::runtime_error("VIDIOC_DQBUF failed: " + std::string(strerror(errno)));

        const uint8_t* raw_data = (const uint8_t*)buffers[buf.index].start;
        size_t raw_len = buf.bytesused;

        std::vector<uint8_t> rgb;
        if (fmt == V4L2_PIX_FMT_MJPEG) {
            rgb = decode_mjpeg(raw_data, raw_len);
        } else {
            rgb = decode_yuyv(raw_data, raw_len);
        }

        // Re-queue buffer immediately (zero-copy: kernel reuses it)
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        ioctl(fd, VIDIOC_QBUF, &buf);

        return rgb;
    }

    // Get actual width/height after negotiation
    int get_width()  const { return width; }
    int get_height() const { return height; }

    const char* format_name() const {
        return (fmt == V4L2_PIX_FMT_MJPEG) ? "MJPEG" : "YUYV";
    }

private:
    bool try_set_format(uint32_t pixel_fmt) {
        v4l2_format vfmt{};
        vfmt.type                  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vfmt.fmt.pix.width         = width;
        vfmt.fmt.pix.height        = height;
        vfmt.fmt.pix.pixelformat   = pixel_fmt;
        vfmt.fmt.pix.field         = V4L2_FIELD_INTERLACED;
        if (ioctl(fd, VIDIOC_S_FMT, &vfmt) < 0) return false;
        width  = vfmt.fmt.pix.width;
        height = vfmt.fmt.pix.height;
        fmt    = vfmt.fmt.pix.pixelformat;
        return true;
    }

    void init_mmap() {
        v4l2_requestbuffers req{};
        req.count  = num_bufs;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0)
            throw std::runtime_error("VIDIOC_REQBUFS failed");
        num_bufs = req.count;
        buffers.resize(num_bufs);

        for (int i = 0; i < num_bufs; i++) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0)
                throw std::runtime_error("VIDIOC_QUERYBUF failed");

            buffers[i].length = buf.length;
            buffers[i].start  = mmap(nullptr, buf.length,
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     fd, buf.m.offset);
            if (buffers[i].start == MAP_FAILED)
                throw std::runtime_error("mmap failed");
        }
    }

    void start_streaming() {
        for (int i = 0; i < num_bufs; i++) {
            v4l2_buffer buf{};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0)
                throw std::runtime_error("VIDIOC_QBUF failed");
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_STREAMON, &type) < 0)
            throw std::runtime_error("VIDIOC_STREAMON failed");
    }

    void stop_streaming() {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd, VIDIOC_STREAMOFF, &type);
    }

    std::vector<uint8_t> decode_mjpeg(const uint8_t* data, size_t len) {
        int w2, h2, ch;
        uint8_t* decoded = stbi_load_from_memory(data, (int)len, &w2, &h2, &ch, 3);
        if (!decoded)
            throw std::runtime_error(std::string("MJPEG decode failed: ") + stbi_failure_reason());
        std::vector<uint8_t> rgb(decoded, decoded + w2 * h2 * 3);
        stbi_image_free(decoded);
        return rgb;
    }

    std::vector<uint8_t> decode_yuyv(const uint8_t* data, size_t /*len*/) {
        // YUYV → RGB conversion
        std::vector<uint8_t> rgb(width * height * 3);
        for (int i = 0; i < width * height / 2; i++) {
            uint8_t y0 = data[4*i+0], u  = data[4*i+1];
            uint8_t y1 = data[4*i+2], v  = data[4*i+3];
            auto yuv2rgb = [](int y, int u, int v, uint8_t* out) {
                int r = y + (int)(1.370705f * (v - 128));
                int g = y - (int)(0.698001f * (v - 128)) - (int)(0.337633f * (u - 128));
                int b = y + (int)(1.732446f * (u - 128));
                out[0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
                out[1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
                out[2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
            };
            yuv2rgb(y0, u, v, &rgb[(2*i+0)*3]);
            yuv2rgb(y1, u, v, &rgb[(2*i+1)*3]);
        }
        return rgb;
    }
};

} // namespace lcv_cam

#else // Non-Linux stub

namespace lcv_cam {
struct Camera {
    Camera(const std::string&, int, int, bool = true) {
        throw std::runtime_error("V4L2 camera not available on this platform (Linux only)");
    }
    std::vector<uint8_t> capture_frame() { return {}; }
    int get_width()  const { return 0; }
    int get_height() const { return 0; }
    const char* format_name() const { return "none"; }
};
} // namespace lcv_cam
#endif // __linux__
