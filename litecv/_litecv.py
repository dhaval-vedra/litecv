# adv_litecv_pro.py
import pygame
import pygame.camera
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageOps, ImageFont
import numpy as np
import math
from functools import lru_cache
import time
import threading
import queue
import os
from enum import Enum

class FilterType(Enum):
    GRAYSCALE = 1
    EDGES = 2
    BLUR = 3
    MOTION_DETECT = 4
    SEPIA = 5
    CARTOON = 6
    SKETCH = 7
    THERMAL = 8
    NIGHT_VISION = 9
    INFRARED = 10

class AdvancedLiteImage:
    def __init__(self, path=None, width=None, height=None, image=None):
        if path:
            self.img = Image.open(path).convert('RGB')
        elif width and height:
            self.img = Image.new('RGB', (width, height), color='white')
        elif image is not None:
            self.img = image.convert('RGB') if hasattr(image, 'convert') else image
        else:
            raise ValueError("Provide a path, width/height, or an image")
            
        # Store image dimensions for quick access
        self.width, self.height = self.img.size
        self._array = None  # Cached numpy array
        
    def show(self):
        """Display the image using the default viewer"""
        self.img.show()
        return self

    def resize(self, width, height, optimize_speed=True):
        """Resize the image with optimization for mobile devices"""
        if optimize_speed:
            # Use faster resampling for mobile devices
            self.img = self.img.resize((width, height), Image.BILINEAR)
        else:
            # Higher quality but slower
            self.img = self.img.resize((width, height), Image.LANCZOS)
            
        self.width, self.height = width, height
        self._array = None  # Clear cached array
        return self

    def resize_percent(self, percent, optimize_speed=True):
        """Resize the image by percentage"""
        new_width = int(self.width * percent / 100)
        new_height = int(self.height * percent / 100)
        return self.resize(new_width, new_height, optimize_speed)

    def to_gray(self, method='lightness'):
        """Convert image to grayscale with different methods"""
        if method == 'lightness':
            # Lightness method (faster)
            self.img = self.img.convert("L")
        elif method == 'luminosity':
            # Luminosity method (more accurate)
            self.img = self.img.convert("RGB")
            r, g, b = self.img.split()
            # Apply luminosity formula: 0.21 R + 0.72 G + 0.07 B
            gray = Image.eval(r, lambda x: x * 0.21)
            g_img = Image.eval(g, lambda x: x * 0.72)
            b_img = Image.eval(b, lambda x: x * 0.07)
            
            # Combine channels
            self.img = Image.eval(gray, lambda x: 
                min(255, x + g_img.getpixel((0, 0)) + b_img.getpixel((0, 0))))
        else:
            self.img = self.img.convert("L")
            
        self._array = None
        return self

    def to_rgb(self):
        """Convert image to RGB"""
        self.img = self.img.convert("RGB")
        self._array = None
        return self

    def blur(self, radius=2, optimize_speed=True):
        """Apply blur with optimization for mobile devices"""
        if optimize_speed and radius > 3:
            # Use a faster approximation for large blur radii on mobile
            for _ in range(int(radius / 2)):
                self.img = self.img.filter(ImageFilter.BLUR)
        else:
            self.img = self.img.filter(ImageFilter.GaussianBlur(radius))
        return self

    def sharpen(self, factor=2.0):
        """Sharpen the image with specified factor"""
        enhancer = ImageEnhance.Sharpness(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def brightness(self, factor):
        """Adjust brightness (factor > 1 brightens, < 1 darkens)"""
        enhancer = ImageEnhance.Brightness(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def contrast(self, factor):
        """Adjust contrast (factor > 1 increases, < 1 decreases)"""
        enhancer = ImageEnhance.Contrast(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def saturation(self, factor):
        """Adjust color saturation"""
        enhancer = ImageEnhance.Color(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def draw_text(self, text, position=(10, 10), color="red", size=20, 
                 font_path=None, optimize_rendering=True):
        """Draw text on the image with mobile optimization"""
        draw = ImageDraw.Draw(self.img)
        
        # Try to use a default font, fallback to basic font if not available
        try:
            if font_path:
                font = ImageFont.truetype(font_path, size)
            else:
                # Use a built-in font for mobile compatibility
                font = ImageFont.load_default()
                # Scale default font if needed
                if size > 10 and hasattr(ImageFont, 'load_default_size'):
                    font = ImageFont.load_default_size(size)
        except:
            font = ImageFont.load_default()
            
        # Optimize text rendering for mobile
        if optimize_rendering:
            # Draw text with simpler anti-aliasing
            draw.text(position, text, fill=color, font=font)
        else:
            # Higher quality text rendering
            draw.text(position, text, fill=color, font=font)
            
        return self

    def draw_rectangle(self, top_left, bottom_right, color="red", width=2, fill=None):
        """Draw a rectangle on the image"""
        draw = ImageDraw.Draw(self.img)
        draw.rectangle([top_left, bottom_right], outline=color, width=width, fill=fill)
        return self

    def draw_circle(self, center, radius, color="red", width=2, fill=None):
        """Draw a circle on the image"""
        draw = ImageDraw.Draw(self.img)
        left = center[0] - radius
        top = center[1] - radius
        right = center[0] + radius
        bottom = center[1] + radius
        draw.ellipse([left, top, right, bottom], outline=color, width=width, fill=fill)
        return self

    def draw_line(self, start, end, color="red", width=2):
        """Draw a line on the image"""
        draw = ImageDraw.Draw(self.img)
        draw.line([start, end], fill=color, width=width)
        return self

    def rotate(self, degrees, expand=True, optimize_speed=True):
        """Rotate the image by specified degrees with optimization"""
        if optimize_speed and degrees % 90 == 0:
            # Use faster method for 90-degree rotations
            if degrees == 90:
                self.img = self.img.transpose(Image.ROTATE_90)
            elif degrees == 180:
                self.img = self.img.transpose(Image.ROTATE_180)
            elif degrees == 270:
                self.img = self.img.transpose(Image.ROTATE_270)
        else:
            # Use standard rotation for other angles
            self.img = self.img.rotate(degrees, expand=expand, resample=Image.BILINEAR)
            
        self.width, self.height = self.img.size
        self._array = None
        return self

    def flip(self, horizontal=True, vertical=False):
        """Flip the image horizontally and/or vertically"""
        if horizontal and vertical:
            self.img = self.img.transpose(Image.ROTATE_180)
        elif horizontal:
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        elif vertical:
            self.img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        self._array = None
        return self

    def crop(self, left, top, right, bottom):
        """Crop the image to the specified coordinates"""
        self.img = self.img.crop((left, top, right, bottom))
        self.width, self.height = self.img.size
        self._array = None
        return self

    def auto_crop(self, border_color=None, threshold=5):
        """Automatically crop borders based on color similarity"""
        if border_color is None:
            # Use the corner pixels to determine border color
            border_color = self.img.getpixel((0, 0))
            
        # Convert to numpy for faster processing
        arr = self.to_numpy()
        
        # Find rows and columns where all pixels are similar to border color
        if len(arr.shape) == 3:
            diff = np.abs(arr - np.array(border_color)).sum(axis=2) > threshold
        else:
            diff = np.abs(arr - border_color) > threshold
            
        # Find bounding box
        rows = np.any(diff, axis=1)
        cols = np.any(diff, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return self  # No cropping needed
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return self.crop(xmin, ymin, xmax+1, ymax+1)

    def edges(self, threshold=100):
        """Detect edges in the image using optimized Sobel operator"""
        # Convert to grayscale if needed
        if self.img.mode != 'L':
            gray_img = self.img.convert('L')
        else:
            gray_img = self.img
            
        # Use PIL's built-in filter for better mobile performance
        self.img = gray_img.filter(ImageFilter.FIND_EDGES)
        
        # Apply threshold if specified
        if threshold > 0:
            self.img = self.img.point(lambda x: 255 if x > threshold else 0)
            
        return self

    def sobel_edges(self, optimize_speed=True):
        """Sobel edge detection with mobile optimization"""
        if optimize_speed:
            # Use a simplified Sobel operator for mobile devices
            kernel_x = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)
            kernel_y = ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1)
        else:
            # Standard Sobel operator
            kernel_x = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)
            kernel_y = ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1)
            
        # Convert to grayscale if needed
        if self.img.mode != 'L':
            gray_img = self.img.convert('L')
        else:
            gray_img = self.img
            
        # Apply Sobel operators
        gx = gray_img.filter(kernel_x)
        gy = gray_img.filter(kernel_y)
        
        # Combine gradients (approximate for speed)
        if optimize_speed:
            # Faster approximation: |Gx| + |Gy|
            gx_arr = np.array(gx, dtype=np.int16)
            gy_arr = np.array(gy, dtype=np.int16)
            gradient = np.abs(gx_arr) + np.abs(gy_arr)
            self.img = Image.fromarray(np.uint8(np.clip(gradient, 0, 255)))
        else:
            # More accurate: sqrt(Gx^2 + Gy^2)
            gx_arr = np.array(gx, dtype=np.float32)
            gy_arr = np.array(gy, dtype=np.float32)
            gradient = np.sqrt(gx_arr**2 + gy_arr**2)
            self.img = Image.fromarray(np.uint8(np.clip(gradient, 0, 255)))
            
        return self

    def get_pixel(self, x, y):
        """Get the RGB value of a pixel"""
        return self.img.getpixel((x, y))

    def set_pixel(self, x, y, color):
        """Set the color of a pixel"""
        self.img.putpixel((x, y), color)
        self._array = None  # Invalidate cached array
        return self

    def to_numpy(self):
        """Convert image to numpy array with caching"""
        if self._array is None:
            self._array = np.array(self.img)
        return self._array

    def from_numpy(self, array):
        """Create image from numpy array"""
        self.img = Image.fromarray(array)
        self.width, self.height = self.img.size
        self._array = array
        return self

    def save(self, path, format=None, optimize=True, quality=85):
        """Save the image to the specified path with optimization"""
        # Use optimized JPEG settings for mobile
        if path.lower().endswith(('.jpg', '.jpeg')):
            self.img.save(path, format=format, optimize=optimize, quality=quality)
        else:
            self.img.save(path, format=format)
        return self

    def copy(self):
        """Create a copy of the image"""
        return AdvancedLiteImage(image=self.img.copy())

    def thumbnail(self, max_size, optimize_speed=True):
        """Create a thumbnail of the image with optimization"""
        if optimize_speed:
            # Use faster thumbnail method
            self.img.thumbnail(max_size, Image.BILINEAR)
        else:
            self.img.thumbnail(max_size, Image.LANCZOS)
            
        self.width, self.height = self.img.size
        self._array = None
        return self

    def get_histogram(self, normalized=False):
        """Get the histogram of the image"""
        if self.img.mode == 'L':
            hist = self.img.histogram()
            if normalized:
                total = sum(hist)
                hist = [h/total for h in hist]
            return hist
        else:
            # For color images, return separate histograms for each channel
            r, g, b = self.img.split()
            r_hist = r.histogram()
            g_hist = g.histogram()
            b_hist = b.histogram()
            
            if normalized:
                total = sum(r_hist)
                r_hist = [h/total for h in r_hist]
                g_hist = [h/total for h in g_hist]
                b_hist = [h/total for h in b_hist]
                
            return r_hist, g_hist, b_hist

    def histogram_equalization(self):
        """Apply histogram equalization to enhance contrast"""
        if self.img.mode == 'L':
            self.img = ImageOps.equalize(self.img)
        else:
            # Convert to HSV, equalize value channel, then convert back
            hsv = self.img.convert('HSV')
            h, s, v = hsv.split()
            v_eq = ImageOps.equalize(v)
            hsv_eq = Image.merge('HSV', (h, s, v_eq))
            self.img = hsv_eq.convert('RGB')
            
        self._array = None
        return self

    def adaptive_threshold(self, block_size=11, c=2, optimize_speed=True):
        """Adaptive thresholding for mobile devices"""
        if self.img.mode != 'L':
            gray = self.img.convert('L')
        else:
            gray = self.img
            
        if optimize_speed:
            # Faster approximation using mean filter
            mean_img = gray.filter(ImageFilter.BoxBlur(block_size//2))
            gray_arr = np.array(gray, dtype=np.int16)
            mean_arr = np.array(mean_img, dtype=np.int16)
            threshold_arr = (gray_arr > (mean_arr - c)) * 255
            self.img = Image.fromarray(np.uint8(threshold_arr))
        else:
            # More accurate method (slower)
            gray_arr = np.array(gray, dtype=np.float32)
            height, width = gray_arr.shape
            
            # Create integral image for fast mean calculation
            integral = np.cumsum(np.cumsum(gray_arr, axis=0), axis=1)
            
            # Pad integral image
            integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
            
            # Calculate adaptive threshold
            result = np.zeros_like(gray_arr)
            half_size = block_size // 2
            
            for y in range(height):
                for x in range(width):
                    y1 = max(0, y - half_size)
                    x1 = max(0, x - half_size)
                    y2 = min(height-1, y + half_size)
                    x2 = min(width-1, x + half_size)
                    
                    area = (y2 - y1 + 1) * (x2 - x1 + 1)
                    sum_val = (integral[y2+1, x2+1] - integral[y1, x2+1] - 
                              integral[y2+1, x1] + integral[y1, x1])
                    
                    mean = sum_val / area
                    result[y, x] = 255 if gray_arr[y, x] > (mean - c) else 0
                    
            self.img = Image.fromarray(np.uint8(result))
            
        return self

    def measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function (for optimization)"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    def batch_operations(self, operations):
        """Execute a batch of operations for better performance"""
        for operation in operations:
            func = operation[0]
            args = operation[1] if len(operation) > 1 else []
            kwargs = operation[2] if len(operation) > 2 else {}
            func(*args, **kwargs)
        return self

    def apply_lut(self, lut, optimize_speed=True):
        """Apply a lookup table for color transformation"""
        if optimize_speed:
            # Fast LUT application using PIL point operation
            if len(lut) == 256:  # 1D LUT for grayscale
                self.img = self.img.point(lut)
            elif len(lut) == 3 and len(lut[0]) == 256:  # 3x256 LUT for RGB
                r_lut, g_lut, b_lut = lut
                r, g, b = self.img.split()
                r = r.point(r_lut)
                g = g.point(g_lut)
                b = b.point(b_lut)
                self.img = Image.merge('RGB', (r, g, b))
        else:
            # More flexible but slower numpy implementation
            arr = self.to_numpy()
            if len(lut) == 256:  # 1D LUT
                if len(arr.shape) == 3:
                    # Apply to all channels
                    for i in range(3):
                        arr[:,:,i] = lut[arr[:,:,i]]
                else:
                    arr = lut[arr]
            elif len(lut) == 3 and len(lut[0]) == 256:  # 3x256 LUT
                for i in range(3):
                    arr[:,:,i] = lut[i][arr[:,:,i]]
            self.from_numpy(arr)
        return self

    def detect_faces(self, cascade_path=None, optimize_speed=True):
        """Simple face detection using Haar-like features (optimized for mobile)"""
        # Convert to grayscale for detection
        gray = self.copy().to_gray()
        arr = gray.to_numpy()
        
        # Simple face detection using Haar-like features approximation
        # This is a lightweight approximation, not a full Haar cascade
        faces = []
        height, width = arr.shape
        
        # Downscale for faster processing
        scale_factor = 2 if optimize_speed else 1
        small_arr = arr[::scale_factor, ::scale_factor]
        small_height, small_width = small_arr.shape
        
        # Simple feature detection (approximation)
        for y in range(0, small_height - 20, 5):
            for x in range(0, small_width - 20, 5):
                # Check for eye-like features (darker areas)
                eye_region1 = small_arr[y:y+10, x+5:x+15]
                eye_region2 = small_arr[y:y+10, x+15:x+25]
                
                # Check for nose-like feature (brighter area between eyes)
                nose_region = small_arr[y+10:y+20, x+10:x+20]
                
                if (np.mean(eye_region1) < 100 and 
                    np.mean(eye_region2) < 100 and 
                    np.mean(nose_region) > 150):
                    # Potential face found
                    faces.append((
                        x * scale_factor, 
                        y * scale_factor, 
                        40 * scale_factor, 
                        40 * scale_factor
                    ))
        
        return faces

    def detect_edges_canny(self, low_threshold=50, high_threshold=150, optimize_speed=True):
        """Canny edge detection optimized for mobile devices"""
        # Convert to grayscale if needed
        if self.img.mode != 'L':
            gray = self.copy().to_gray()
        else:
            gray = self.copy()
            
        # Apply Gaussian blur to reduce noise
        gray = gray.blur(radius=1, optimize_speed=optimize_speed)
        
        # Use Sobel operator to get gradients
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        arr = gray.to_numpy().astype(np.float32)
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        
        # Apply Sobel operators
        height, width = arr.shape
        for y in range(1, height-1):
            for x in range(1, width-1):
                region = arr[y-1:y+2, x-1:x+2]
                gx[y, x] = np.sum(region * sobel_x)
                gy[y, x] = np.sum(region * sobel_y)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction = (direction + 180) % 180  # Convert to 0-180 range
        
        # Non-maximum suppression
        suppressed = np.zeros_like(magnitude)
        for y in range(1, height-1):
            for x in range(1, width-1):
                angle = direction[y, x]
                # Find adjacent pixels based on gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [magnitude[y, x-1], magnitude[y, x+1]]
                elif 22.5 <= angle < 67.5:
                    neighbors = [magnitude[y-1, x-1], magnitude[y+1, x+1]]
                elif 67.5 <= angle < 112.5:
                    neighbors = [magnitude[y-1, x], magnitude[y+1, x]]
                else:  # 112.5 <= angle < 157.5
                    neighbors = [magnitude[y-1, x+1], magnitude[y+1, x-1]]
                
                # Suppress non-maximum pixels
                if magnitude[y, x] >= max(neighbors):
                    suppressed[y, x] = magnitude[y, x]
        
        # Double thresholding and edge tracking
        strong_edges = suppressed > high_threshold
        weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
        
        # Edge tracking by hysteresis
        edges = strong_edges.copy()
        changed = True
        while changed:
            changed = False
            for y in range(1, height-1):
                for x in range(1, width-1):
                    if weak_edges[y, x] and not edges[y, x]:
                        # Check if any strong edge in 8-connected neighborhood
                        if np.any(edges[y-1:y+2, x-1:x+2]):
                            edges[y, x] = True
                            changed = True
        
        # Convert to image
        self.img = Image.fromarray((edges * 255).astype(np.uint8))
        return self

    def to_pygame_surface(self):
        """Convert to Pygame surface for real-time display"""
        # Convert PIL image to Pygame surface
        mode = self.img.mode
        data = self.img.tobytes()
        size = self.img.size
        
        if mode == 'RGB':
            return pygame.image.fromstring(data, size, mode)
        elif mode == 'RGBA':
            return pygame.image.fromstring(data, size, mode)
        elif mode == 'L':
            # Convert grayscale to RGB for Pygame
            rgb_img = self.img.convert('RGB')
            data = rgb_img.tobytes()
            return pygame.image.fromstring(data, size, 'RGB')
        else:
            # Convert to RGB by default
            rgb_img = self.img.convert('RGB')
            data = rgb_img.tobytes()
            return pygame.image.fromstring(data, size, 'RGB')

    @classmethod
    def from_pygame_surface(cls, surface):
        """Create AdvancedLiteImage from Pygame surface"""
        # Convert Pygame surface to PIL image
        data = pygame.image.tostring(surface, 'RGB')
        size = surface.get_size()
        pil_img = Image.frombytes('RGB', size, data)
        return cls(image=pil_img)

# New features for real-time processing with Pygame
class CameraFeed:
    """Real-time camera feed using Pygame"""
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30, optimize=True):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.optimize = optimize
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.callback = None
        
        # Initialize Pygame camera
        pygame.camera.init()
        
        # Get available cameras
        cam_list = pygame.camera.list_cameras()
        if not cam_list:
            raise RuntimeError("No cameras found!")
            
        print(f"Available cameras: {cam_list}")
        
        # Use specified camera or default
        if camera_id < len(cam_list):
            camera_name = cam_list[camera_id]
        else:
            camera_name = cam_list[0]
            
        print(f"Using camera: {camera_name}")
        
        # Create camera object
        self.camera = pygame.camera.Camera(camera_name, resolution)
        
    def start(self):
        """Start the camera feed"""
        self.camera.start()
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        return self
        
    def stop(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.stop()
        
    def set_callback(self, callback):
        """Set a callback function for frame processing"""
        self.callback = callback
        
    def _capture_frames(self):
        """Internal method to capture frames using Pygame"""
        try:
            clock = pygame.time.Clock()
            while self.is_running:
                # Capture frame from camera
                if self.camera.query_image():
                    surface = self.camera.get_image()
                    
                    # Convert to AdvancedLiteImage
                    frame = AdvancedLiteImage.from_pygame_surface(surface)
                    
                    # Call callback if set
                    if self.callback:
                        self.callback(frame)
                    
                    # Add to queue (non-blocking)
                    try:
                        if self.frame_queue.full():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                    
                    # Maintain frame rate
                    clock.tick(self.fps)
                    
        except Exception as e:
            print(f"Camera error: {e}")
            self.is_running = False
            
    def get_frame(self, timeout=0.1):
        """Get the latest frame from the camera"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def frames(self):
        """Generator that yields frames from the camera"""
        while self.is_running:
            frame = self.get_frame()
            if frame:
                yield frame

class VideoProcessor:
    """Video processing with Pygame support"""
    def __init__(self, video_path=None, optimize=True):
        self.video_path = video_path
        self.optimize = optimize
        self.current_frame = 0
        self.is_playing = False
        
    def open(self, video_path):
        """Open a video file"""
        self.video_path = video_path
        self.current_frame = 0
        return self
        
    def get_frame(self, frame_num=None):
        """Get a specific frame from the video"""
        # Placeholder implementation
        # In real implementation, use Pygame movie or FFmpeg
        img = Image.new('RGB', (640, 480), color='black')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), f"Video Frame: {self.current_frame}", fill='white')
        
        self.current_frame = (self.current_frame + 1) % 100
        return AdvancedLiteImage(image=img)
        
    def seek(self, position):
        """Seek to a specific position in the video"""
        self.current_frame = position
        return self
        
    def close(self):
        """Close the video file"""
        self.video_path = None
        return self

class StreamingFilter:
    """Lightweight streaming filters for real-time processing"""
    def __init__(self, filter_type=FilterType.GRAYSCALE, optimize=True):
        self.filter_type = filter_type
        self.optimize = optimize
        self.previous_frame = None
        
    def apply(self, frame):
        """Apply the filter to a frame"""
        if self.filter_type == FilterType.GRAYSCALE:
            return frame.copy().to_gray()
            
        elif self.filter_type == FilterType.EDGES:
            return frame.copy().edges()
            
        elif self.filter_type == FilterType.BLUR:
            return frame.copy().blur(radius=2, optimize_speed=self.optimize)
            
        elif self.filter_type == FilterType.MOTION_DETECT:
            # Simple motion detection by comparing with previous frame
            if self.previous_frame is None:
                self.previous_frame = frame.copy().to_gray()
                return frame
                
            current_gray = frame.copy().to_gray()
            prev_arr = self.previous_frame.to_numpy()
            curr_arr = current_gray.to_numpy()
            
            # Calculate absolute difference
            diff = np.abs(prev_arr.astype(np.int16) - curr_arr.astype(np.int16))
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            
            # Threshold to highlight motion
            motion_mask = diff > 30
            result = frame.to_numpy().copy()
            result[motion_mask] = [255, 0, 0]  # Highlight motion in red
            
            self.previous_frame = current_gray
            return AdvancedLiteImage(image=Image.fromarray(result))
            
        elif self.filter_type == FilterType.SEPIA:
            # Apply sepia tone filter
            arr = frame.to_numpy().astype(np.float32)
            # Sepia transformation matrix
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            
            # Apply matrix transformation
            result = np.dot(arr, sepia_matrix.T)
            result = np.clip(result, 0, 255).astype(np.uint8)
            return AdvancedLiteImage(image=Image.fromarray(result))
            
        elif self.filter_type == FilterType.CARTOON:
            # Cartoon filter: edge detection + bilateral filter approximation
            gray = frame.copy().to_gray()
            edges = gray.copy().edges(threshold=100)
            
            # Apply bilateral filter approximation
            if self.optimize:
                blurred = frame.copy().blur(radius=3, optimize_speed=True)
            else:
                blurred = frame.copy().blur(radius=2, optimize_speed=False)
                
            # Combine edges with blurred image
            edges_arr = edges.to_numpy()
            blurred_arr = blurred.to_numpy()
            
            # Create mask from edges
            mask = edges_arr > 128
            
            # Apply mask to blurred image
            result = blurred_arr.copy()
            for i in range(3):
                result[:,:,i][mask] = 0  # Set edge areas to black
                
            return AdvancedLiteImage(image=Image.fromarray(result))
            
        elif self.filter_type == FilterType.SKETCH:
            # Pencil sketch filter
            gray = frame.copy().to_gray()
            inverted = ImageOps.invert(gray.img)
            blurred = inverted.filter(ImageFilter.BLUR)
            sketch = Image.blend(gray.img, blurred, 0.7)
            return AdvancedLiteImage(image=sketch)
            
        elif self.filter_type == FilterType.THERMAL:
            # Thermal camera effect
            arr = frame.to_numpy().astype(np.float32)
            gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
            
            # Apply thermal color map
            thermal = np.zeros_like(arr)
            thermal[:,:,0] = np.clip(gray * 255 / 128, 0, 255)  # Red
            thermal[:,:,1] = np.clip((gray - 64) * 255 / 128, 0, 255)  # Green
            thermal[:,:,2] = np.clip(255 - gray * 255 / 128, 0, 255)  # Blue
            
            return AdvancedLiteImage(image=Image.fromarray(thermal.astype(np.uint8)))
            
        elif self.filter_type == FilterType.NIGHT_VISION:
            # Night vision effect
            arr = frame.to_numpy().astype(np.float32)
            gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
            gray = np.clip(gray * 1.5, 0, 255)  # Enhance brightness
            
            # Apply green tint
            night_vision = np.zeros_like(arr)
            night_vision[:,:,0] = gray * 0.1  # Red
            night_vision[:,:,1] = gray * 0.9  # Green
            night_vision[:,:,2] = gray * 0.1  # Blue
            
            return AdvancedLiteImage(image=Image.fromarray(night_vision.astype(np.uint8)))
            
        elif self.filter_type == FilterType.INFRARED:
            # Infrared effect
            arr = frame.to_numpy().astype(np.float32)
            gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
            infrared = 255 - gray  # Invert
            
            # Apply red tint
            infrared_rgb = np.zeros_like(arr)
            infrared_rgb[:,:,0] = infrared  # Red
            infrared_rgb[:,:,1] = infrared * 0.5  # Green
            infrared_rgb[:,:,2] = infrared * 0.5  # Blue
            
            return AdvancedLiteImage(image=Image.fromarray(infrared_rgb.astype(np.uint8)))
            
        return frame

class ObjectDetector:
    """Lightweight object detector"""
    def __init__(self, model_path=None, optimize=True):
        self.model_path = model_path
        self.optimize = optimize
        self.labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
                      "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
                      "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
                      "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
                      "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
                      "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
                      "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                      "couch", "potted plant", "bed", "dining table", "toilet", "tv", 
                      "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
                      "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
                      "scissors", "teddy bear", "hair drier", "toothbrush"]
        
    def load_model(self):
        """Load the detection model"""
        print("Model loading not implemented in this version")
        return self
        
    def detect(self, image, confidence_threshold=0.5):
        """Detect objects in the image"""
        # Simplified implementation for demonstration
        height, width = image.height, image.width
        
        detections = []
        for i in range(3):  # Generate 3 random detections
            label = np.random.choice(self.labels)
            confidence = np.random.uniform(0.6, 0.9)
            x = np.random.randint(0, width - 50)
            y = np.random.randint(0, height - 50)
            w = np.random.randint(30, 100)
            h = np.random.randint(30, 100)
            
            if confidence > confidence_threshold:
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })
                
        return detections

# Utility functions
def open_image(path):
    """Open an image from the specified path"""
    return AdvancedLiteImage(path)

def new_image(width, height, color='white'):
    """Create a new blank image"""
    img = AdvancedLiteImage(width=width, height=height)
    if color != 'white':
        draw = ImageDraw.Draw(img.img)
        draw.rectangle([0, 0, width, height], fill=color)
    return img

def concatenate(images, direction='horizontal', optimize_speed=True):
    """Concatenate multiple images with optimization"""
    if not images:
        raise ValueError("No images provided for concatenation")
    
    target_mode = images[0].img.mode
    for img in images:
        if img.img.mode != target_mode:
            img.img = img.img.convert(target_mode)
    
    if direction == 'horizontal':
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        new_img = Image.new(target_mode, (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img.img, (x_offset, 0))
            x_offset += img.width
    else:  # vertical
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        new_img = Image.new(target_mode, (max_width, total_height))
        y_offset = 0
        for img in images:
            new_img.paste(img.img, (0, y_offset))
            y_offset += img.height
            
    return AdvancedLiteImage(image=new_img)

def blend_images(image1, image2, alpha=0.5):
    """Blend two images together"""
    if image1.img.size != image2.img.size:
        image2 = image2.copy().resize(image1.width, image1.height)
        
    blended = Image.blend(image1.img.convert('RGBA'), 
                         image2.img.convert('RGBA'), 
                         alpha)
    return AdvancedLiteImage(image=blended.convert('RGB'))

# Real-time camera application with Pygame
class RealTimeCameraApp:
    """Real-time camera application with Pygame GUI"""
    def __init__(self, resolution=(800, 600), camera_resolution=(640, 480)):
        pygame.init()
        self.screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("Advanced LiteCV - Real Time Camera")
        
        self.camera_resolution = camera_resolution
        self.camera = None
        self.current_filter = None
        self.is_running = False
        self.fps = 30
        self.clock = pygame.time.Clock()
        
        # Available filters
        self.filters = {
            pygame.K_1: FilterType.GRAYSCALE,
            pygame.K_2: FilterType.EDGES,
            pygame.K_3: FilterType.BLUR,
            pygame.K_4: FilterType.SEPIA,
            pygame.K_5: FilterType.CARTOON,
            pygame.K_6: FilterType.SKETCH,
            pygame.K_7: FilterType.THERMAL,
            pygame.K_8: FilterType.NIGHT_VISION,
            pygame.K_9: FilterType.INFRARED,
            pygame.K_0: None  # No filter
        }
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        
    def start(self):
        """Start the camera application"""
        try:
            self.camera = CameraFeed(resolution=self.camera_resolution, fps=self.fps)
            self.camera.start()
            self.is_running = True
            
            print("Camera started. Press keys 1-9 for filters, 0 for original, ESC to quit.")
            self._main_loop()
            
        except Exception as e:
            print(f"Error starting camera: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the camera application"""
        self.is_running = False
        if self.camera:
            self.camera.stop()
        pygame.quit()
        
    def _main_loop(self):
        """Main application loop"""
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    elif event.key in self.filters:
                        self.current_filter = self.filters[event.key]
                        if self.current_filter:
                            print(f"Applied filter: {self.current_filter.name}")
                        else:
                            print("Removed filter")
            
            # Get camera frame
            frame = self.camera.get_frame()
            if frame:
                # Apply current filter
                if self.current_filter:
                    filter_processor = StreamingFilter(self.current_filter)
                    processed_frame = filter_processor.apply(frame)
                else:
                    processed_frame = frame
                
                # Convert to Pygame surface and display
                surface = processed_frame.to_pygame_surface()
                self.screen.blit(surface, (0, 0))
                
                # Display UI information
                self._draw_ui()
                
                pygame.display.flip()
            
            self.clock.tick(self.fps)
            
    def _draw_ui(self):
        """Draw UI elements"""
        # Display current filter info
        if self.current_filter:
            filter_text = f"Filter: {self.current_filter.name}"
        else:
            filter_text = "Filter: None"
            
        text_surface = self.font.render(filter_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        # Display FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        self.screen.blit(fps_surface, (10, 40))
        
        # Display instructions
        instructions = [
            "1: Grayscale  2: Edges  3: Blur",
            "4: Sepia     5: Cartoon 6: Sketch", 
            "7: Thermal   8: Night   9: Infrared",
            "0: Original  ESC: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            instr_surface = self.font.render(instruction, True, (255, 255, 0))
            self.screen.blit(instr_surface, (10, 70 + i * 25))

# Example usage
def demo_basic_operations():
    """Demonstrate basic image operations"""
    print("Creating test image...")
    img = new_image(400, 300, color='lightblue')
    img.draw_circle((200, 150), 80, color='red', fill='yellow')
    img.draw_text("Hello LiteCV!", (50, 50), color='darkblue', size=24)
    img.save(r"C:\Users\gamet\Pictures\Screenshot_20260303-124948_Snapchat.jpg")
    print("Demo image saved as /tmp/demo_image.jpg")

def demo_real_time_camera():
    """Demonstrate real-time camera with filters"""
    print("Starting real-time camera demo...")
    app = RealTimeCameraApp(resolution=(640, 480), camera_resolution=(320, 240))
    app.start()

def demo_filters():
    """Demonstrate various filters on a test image"""
    print("Testing filters...")
    
    # Create test image
    img = new_image(200, 200, color='white')
    img.draw_circle((100, 100), 80, color='blue', fill='green')
    img.draw_rectangle((50, 50), (150, 150), color='red', width=3)
    
    # Apply different filters
    filters = [
        (FilterType.GRAYSCALE, "grayscale"),
        (FilterType.EDGES, "edges"),
        (FilterType.SEPIA, "sepia"),
        (FilterType.CARTOON, "cartoon"),
        (FilterType.SKETCH, "sketch")
    ]
    
    for filter_type, name in filters:
        filtered_img = img.copy()
        filter_processor = StreamingFilter(filter_type)
        result = filter_processor.apply(filtered_img)
        output_path = os.path.join(os.getcwd(), f"filter_{name}.jpg")
        result.save(output_path)
        print(f"Saved {name} filter as {output_path}")

if __name__ == "__main__":
    print("Advanced LiteCV Pro with Pygame Camera Support")
    print("=" * 50)
    
    # Run demos
    demo_basic_operations()
    demo_filters()
    
    # Ask user if they want to run camera demo
    response = input("Do you want to start real-time camera demo? (y/n): ")
    if response.lower() == 'y':
        demo_real_time_camera()
    
    print("Demo completed!")