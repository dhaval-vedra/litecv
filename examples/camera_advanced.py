"""
camera_advanced.py — Advanced camera usage with async + profiling + streaming

Demonstrates:
  - Async camera (non-blocking)
  - Frame buffer (temporal ops)
  - Profiler (FPS, per-step latency)
  - MJPEG streaming server

Run on Raspberry Pi with a USB or Pi Camera:
    python examples/camera_advanced.py --device /dev/video0 --stream

Requirements: litecv-advanced + C++ extension compiled
"""
import sys
import os
import argparse
import time
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import litecv
from litecv import (
    CameraFeed, AsyncCamera, FrameBuffer,
    Profiler, fps_counter, MemoryMonitor,
    MJPEGServer, MotionDetector, pipeline as P
)


def parse_args():
    p = argparse.ArgumentParser(description="litecv-advanced camera demo")
    p.add_argument("--device",  default="/dev/video0")
    p.add_argument("--width",   type=int, default=640)
    p.add_argument("--height",  type=int, default=480)
    p.add_argument("--frames",  type=int, default=100, help="Frames to capture (0=infinite)")
    p.add_argument("--stream",  action="store_true", help="Start MJPEG HTTP stream")
    p.add_argument("--port",    type=int, default=8080)
    p.add_argument("--effect",  default="clahe",
                   choices=["none","clahe","sepia","thermal","night","sketch","cartoon"])
    p.add_argument("--motion",  action="store_true", help="Enable motion detection")
    p.add_argument("--save",    default="", help="Save frames to directory")
    p.add_argument("--async_cam", action="store_true", help="Use async camera")
    return p.parse_args()


def get_effect(name):
    effects = {
        "none":    lambda img: img,
        "clahe":   P.clahe(clip_limit=3.0),
        "sepia":   P.sepia(),
        "thermal": P.thermal(),
        "night":   P.night_vision(),
        "sketch":  P.sketch(),
        "cartoon": P.cartoon(),
    }
    return effects.get(name, lambda img: img)


def demo_sync_camera(args):
    """Synchronous camera with profiler."""
    print(f"\n[Sync Camera] device={args.device} effect={args.effect}")

    prof   = Profiler()
    fps    = fps_counter(window=30)
    mem    = MemoryMonitor()
    motion = MotionDetector(threshold=25, min_area=1000) if args.motion else None
    effect = get_effect(args.effect)
    saver  = P.save_dir(args.save) if args.save else None
    stream = None

    if args.stream:
        stream = MJPEGServer(port=args.port, quality=70).start()
        print(f"  Stream: {stream.url}")

    frame_count = 0
    stop = [False]

    def on_signal(*_): stop[0] = True
    signal.signal(signal.SIGINT, on_signal)

    with CameraFeed(args.device, args.width, args.height) as cam:
        while not stop[0]:
            if args.frames > 0 and frame_count >= args.frames:
                break

            with prof.step("capture"):
                frame = cam.capture()
            prof.frame()

            with prof.step("effect"):
                processed = effect(frame) if callable(effect) else frame

            if motion:
                with prof.step("motion"):
                    detections = motion.detect(frame)
                    for d in detections:
                        processed = processed.draw_rect(
                            d.x, d.y, d.w, d.h,
                            color=(255, 0, 0), thickness=2)
                        processed = processed.draw_text(
                            f"Motion {d.score:.0f}",
                            d.x, max(0, d.y-20), (255,0,0))

            if stream:
                with prof.step("stream"):
                    stream.push_frame(processed)

            if saver:
                with prof.step("save"):
                    saver(processed)

            frame_count += 1
            current_fps = fps()
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count:4d} | FPS: {current_fps:5.1f} | "
                      f"RSS: {mem.rss_mb:.1f}MB", end="\r")

    print()
    prof.report()
    if stream:
        stream.stop()
        print(f"\nStream stats: {stream.stats}")


def demo_async_camera(args):
    """Non-blocking async camera."""
    print(f"\n[Async Camera] device={args.device} buffer=3")
    fps = fps_counter(window=30)
    effect = get_effect(args.effect)
    stop = [False]

    def on_signal(*_): stop[0] = True
    signal.signal(signal.SIGINT, on_signal)

    with AsyncCamera(args.device, args.width, args.height, buffer_size=3) as cam:
        print(f"  Camera: {cam.width}×{cam.height}")
        stream = MJPEGServer(port=args.port, quality=70).start() if args.stream else None

        frame_count = 0
        while not stop[0]:
            if args.frames > 0 and frame_count >= args.frames:
                break

            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            processed = effect(frame) if callable(effect) else frame

            if stream:
                stream.push_frame(processed)

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count:4d} | Camera FPS: {cam.fps:.1f} | "
                      f"Processing FPS: {fps():.1f}", end="\r")

        if stream:
            stream.stop()
    print(f"\n  Total frames processed: {frame_count}")


def demo_frame_buffer(args):
    """Temporal operations using FrameBuffer."""
    print(f"\n[Frame Buffer] Temporal motion mask")
    buf = FrameBuffer(capacity=30)

    with CameraFeed(args.device, args.width, args.height) as cam:
        for i in range(args.frames or 50):
            frame = cam.capture()
            buf.push(frame)

            if buf.full:
                motion_mask = buf.motion_mask(threshold=20)
                avg_frame   = buf.average()
                if motion_mask:
                    motion_pixels = (motion_mask._data > 0).sum()
                    print(f"  Frame {i}: motion pixels = {motion_pixels}", end="\r")
                if avg_frame and args.save:
                    avg_frame.save(f"{args.save}/avg_{i:04d}.jpg")

    print("\n  FrameBuffer demo complete")


def main():
    args = parse_args()
    print(litecv.backend_info())

    if args.async_cam:
        demo_async_camera(args)
    elif args.frames > 0 and args.frames <= 20:
        demo_frame_buffer(args)
    else:
        demo_sync_camera(args)


if __name__ == "__main__":
    main()
