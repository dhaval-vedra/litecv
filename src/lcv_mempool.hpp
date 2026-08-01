/*
 * lcv_mempool.hpp — Thread-local memory pool for zero-allocation frame processing
 *
 * Edge devices se baar baar malloc/free karna slow hota hai.
 * Ye pool ek hi baar memory allocate karta hai aur reuse karta hai.
 */
#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>

namespace lcv_pool {

// ── Fixed-size block pool ─────────────────────────────────────────────────────
// Allocate blocks of any size up to 'Capacity' bytes.
// Thread-local by design — no locking needed for single-threaded camera pipelines.
template<size_t Capacity>
class BlockPool {
public:
    uint8_t buf[Capacity];
    size_t  used = 0;

    // Bump allocator — resets each frame
    uint8_t* alloc(size_t bytes, size_t align = 64) {
        size_t offset = (used + align - 1) & ~(align - 1);
        if (offset + bytes > Capacity) return nullptr; // overflow: caller falls back to heap
        used = offset + bytes;
        return buf + offset;
    }

    void reset() { used = 0; }

    float utilization() const { return (float)used / (float)Capacity; }
};

// ── Global per-thread pools ───────────────────────────────────────────────────
// Frame pool: up to 4K×4K×3 = ~48 MB. Enough for any Pi camera resolution.
constexpr size_t FRAME_POOL_SIZE = 50 * 1024 * 1024; // 50 MB
thread_local BlockPool<FRAME_POOL_SIZE> g_frame_pool;

// Scratch pool: for intermediate buffers (histograms, kernels, etc.)
constexpr size_t SCRATCH_POOL_SIZE = 8 * 1024 * 1024; // 8 MB
thread_local BlockPool<SCRATCH_POOL_SIZE> g_scratch_pool;

// ── RAII frame reset ──────────────────────────────────────────────────────────
struct FrameScope {
    size_t saved;
    FrameScope() : saved(g_frame_pool.used) {}
    ~FrameScope() { g_frame_pool.used = saved; }
};

// ── Helper: allocate from frame pool, fallback to heap ───────────────────────
inline uint8_t* frame_alloc(size_t bytes) {
    uint8_t* p = g_frame_pool.alloc(bytes);
    if (!p) {
        // Pool exhausted — heap fallback (rare, only for huge images)
        p = new uint8_t[bytes];
    }
    return p;
}

inline uint8_t* scratch_alloc(size_t bytes) {
    uint8_t* p = g_scratch_pool.alloc(bytes);
    if (!p) p = new uint8_t[bytes];
    return p;
}

inline void frame_reset()   { g_frame_pool.reset(); }
inline void scratch_reset() { g_scratch_pool.reset(); }

} // namespace lcv_pool
