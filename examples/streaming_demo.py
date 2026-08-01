"""
streaming_demo.py — MJPEG streaming demo

Stream processed camera frames over HTTP.
Open http://raspberrypi:8080/ in any browser to watch live.

Usage:
    python examples/streaming_demo.py
    python examples/streaming_demo.py --device /dev/video0 --effect night --port 8080
"""
import sys, os, time, argparse, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import litecv
from litecv import from_numpy, new_image, MJPEGServer, fps_counter

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--port",   type=int, default=8080)
    p.add_argument("--effect", default="thermal",
                   choices=["none","clahe","sepia","thermal","night","sketch","cartoon"])
    p.add_argument("--no-camera", action="store_true", help="Use synthetic frames (testing)")
    return p.parse_args()


def synthetic_frame_generator(w=640, h=480):
    """Generate animated synthetic frames for testing without a camera."""
    t = 0
    while True:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        # Animated gradient
        for y in range(h):
            r = int((y / h) * 255)
            g = int(abs(np.sin(t + y * 0.02)) * 255)
            b = int((1 - y / h) * 255)
            arr[y, :] = [r, g, b]
        # Moving ball
        bx = int(w/2 + np.cos(t) * w/3)
        by = int(h/2 + np.sin(t*1.3) * h/3)
        cy, cx = np.ogrid[:h, :w]
        mask = (cx-bx)**2 + (cy-by)**2 <= 30**2
        arr[mask] = [255, 255, 0]
        t += 0.1
        yield from_numpy(arr)


def apply_effect(img, effect_name):
    effects = {
        "none":    lambda i: i,
        "clahe":   lambda i: i.clahe(clip_limit=3.0),
        "sepia":   lambda i: i.sepia(),
        "thermal": lambda i: i.thermal(),
        "night":   lambda i: i.night_vision(),
        "sketch":  lambda i: i.sketch(),
        "cartoon": lambda i: i.cartoon(),
    }
    return effects.get(effect_name, lambda i: i)(img)


def main():
    args = parse_args()
    print(litecv.backend_info())

    fps = fps_counter(30)
    stop = [False]
    def on_signal(*_): stop[0] = True
    signal.signal(signal.SIGINT, on_signal)

    with MJPEGServer(port=args.port, quality=70, max_fps=30) as server:
        print(f"\nStream URL: {server.url}")
        print(f"Snapshot:   http://0.0.0.0:{args.port}/frame.jpg")
        print(f"Status:     http://0.0.0.0:{args.port}/")
        print(f"Effect: {args.effect}")
        print("Press Ctrl+C to stop\n")

        if args.no_camera:
            print("[Using synthetic frames — no camera needed]")
            frame_gen = synthetic_frame_generator()
            frame_count = 0
            while not stop[0]:
                raw = next(frame_gen)
                processed = apply_effect(raw, args.effect)
                server.push_frame(processed)
                frame_count += 1
                current_fps = fps()
                if frame_count % 30 == 0:
                    print(f"  Frames: {frame_count:5d} | FPS: {current_fps:5.1f} | "
                          f"Stats: {server.stats}", end="\r")
                time.sleep(1/30)
        else:
            try:
                from litecv import CameraFeed
                cam = CameraFeed(args.device)
                cam.open()
                print(f"[Camera: {cam.width}×{cam.height} {cam.backend}]")

                def process_fn(img):
                    return apply_effect(img, args.effect)

                server.start_feed(cam, process_fn=process_fn, max_fps=30)

                while not stop[0]:
                    time.sleep(0.5)
                    stats = server.stats
                    print(f"  FPS: {stats['fps']:5.1f} | Frames: {stats['frames_sent']}", end="\r")

                cam.close()
            except Exception as e:
                print(f"Camera error: {e}")
                print("Falling back to synthetic frames…")
                frame_gen = synthetic_frame_generator()
                while not stop[0]:
                    raw = next(frame_gen)
                    server.push_frame(apply_effect(raw, args.effect))
                    time.sleep(1/30)

    print(f"\nFinal stats: {server.stats}")


if __name__ == "__main__":
    main()
