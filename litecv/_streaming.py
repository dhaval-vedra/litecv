"""
litecv._streaming — MJPEG HTTP streaming server

Stream your camera feed over the network — view on any browser,
VLC, or any MJPEG-compatible client (like OpenCV VideoCapture).

Zero extra dependencies — uses Python stdlib http.server only.

Usage:
    from litecv import CameraFeed
    from litecv.streaming import MJPEGServer

    cam = CameraFeed("/dev/video0", 640, 480)
    cam.open()

    with MJPEGServer(port=8080) as server:
        server.start_feed(cam)     # runs camera loop in background
        print("Stream: http://raspberrypi:8080/stream")
        input("Press Enter to stop...")

    # Or stream any frame sequence:
    server = MJPEGServer(port=8080)
    server.start()
    while True:
        frame = cam.capture()
        processed = frame.night_vision()
        server.push_frame(processed)
"""
from __future__ import annotations

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Callable

from ._core import LiteImage


class MJPEGServer:
    """
    Minimal MJPEG streaming server over HTTP.

    View stream at: http://<device-ip>:<port>/stream
    View single frame: http://<device-ip>:<port>/frame.jpg
    View status: http://<device-ip>:<port>/status
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080,
                 quality: int = 70, max_fps: float = 30.0):
        self.host = host
        self.port = port
        self.quality = quality
        self.min_interval = 1.0 / max_fps
        self._frame_bytes: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._feed_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_count = 0
        self._start_time = None

    def push_frame(self, img: LiteImage, quality: int = None):
        """Push a new frame to all connected clients."""
        q = quality or self.quality
        if hasattr(img, 'to_bytes'):
            data = img.to_bytes(".jpg")
        else:
            import io as _io
            try:
                from PIL import Image as PILImage
                buf = _io.BytesIO()
                pil = PILImage.fromarray(img._data, "RGB" if not img.is_gray else "L")
                pil.save(buf, format="JPEG", quality=q)
                data = buf.getvalue()
            except ImportError:
                return
        with self._frame_lock:
            self._frame_bytes = data
            self._frame_count += 1
        self._frame_event.set()
        self._frame_event.clear()

    def start(self) -> "MJPEGServer":
        """Start the HTTP server (non-blocking)."""
        self._running = True
        self._start_time = time.time()

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress default access logs

            def do_GET(self):
                if self.path in ("/stream", "/video"):
                    self._stream()
                elif self.path in ("/frame.jpg", "/snapshot"):
                    self._snapshot()
                elif self.path in ("/status", "/"):
                    self._status()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=--litecv_boundary")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last_t = 0
                try:
                    while server_ref._running:
                        t = time.time()
                        if t - last_t < server_ref.min_interval:
                            time.sleep(0.005)
                            continue
                        with server_ref._frame_lock:
                            data = server_ref._frame_bytes
                        if data is None:
                            time.sleep(0.01)
                            continue
                        try:
                            self.wfile.write(
                                b"--litecv_boundary\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(data)}\r\n\r\n".encode()
                                + data + b"\r\n"
                            )
                            self.wfile.flush()
                            last_t = t
                        except (BrokenPipeError, ConnectionResetError):
                            break
                except Exception:
                    pass

            def _snapshot(self):
                with server_ref._frame_lock:
                    data = server_ref._frame_bytes
                if data is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _status(self):
                uptime = time.time() - (server_ref._start_time or time.time())
                fps = server_ref._frame_count / uptime if uptime > 0 else 0
                html = f"""<!DOCTYPE html>
<html><head><title>litecv MJPEG Stream</title>
<style>body{{font-family:monospace;background:#111;color:#0f0;padding:20px}}
img{{border:2px solid #0f0;max-width:100%}}</style></head>
<body>
<h2>🎥 litecv-advanced MJPEG Stream</h2>
<img src="/stream" alt="Live stream" /><br>
<p>FPS: {fps:.1f} | Frames: {server_ref._frame_count} | Uptime: {uptime:.0f}s</p>
<p>Stream URL: <a href="/stream" style="color:#0ff">/stream</a> |
   Snapshot: <a href="/frame.jpg" style="color:#0ff">/frame.jpg</a></p>
</body></html>"""
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer((self.host, self.port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
            name="litecv-mjpeg")
        self._server_thread.start()
        print(f"[litecv] MJPEG stream: http://{self.host}:{self.port}/")
        print(f"[litecv] Stream URL:   http://{self.host}:{self.port}/stream")
        return self

    def start_feed(self, camera, process_fn: Callable = None,
                   max_fps: float = None) -> "MJPEGServer":
        """
        Start capturing from a CameraFeed in background and push to stream.

        process_fn: optional function LiteImage → LiteImage (apply effects)
        """
        if not self._running:
            self.start()
        interval = 1.0 / (max_fps or 1.0/self.min_interval)

        def _feed_loop():
            while self._running:
                try:
                    frame = camera.capture()
                    if process_fn:
                        frame = process_fn(frame)
                    self.push_frame(frame)
                    time.sleep(interval)
                except Exception as e:
                    print(f"[litecv] Stream feed error: {e}")
                    time.sleep(0.5)

        self._feed_thread = threading.Thread(target=_feed_loop, daemon=True,
                                              name="litecv-feed")
        self._feed_thread.start()
        return self

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()
        if self._server_thread:
            self._server_thread.join(timeout=2.0)

    def __enter__(self) -> "MJPEGServer":
        return self.start()

    def __exit__(self, *args):
        self.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/stream"

    @property
    def stats(self) -> dict:
        uptime = time.time() - (self._start_time or time.time())
        return {
            "frames_sent": self._frame_count,
            "uptime_s": uptime,
            "fps": self._frame_count / uptime if uptime > 0 else 0,
            "url": self.url,
        }
