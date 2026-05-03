"""LiteCV - lightweight computer vision library."""

import os
from ._litecv import (
    AdvancedLiteImage,
    CameraFeed,
    VideoProcessor,
    StreamingFilter,
    ObjectDetector,
    RealTimeCameraApp,
    FilterType,
    open_image,
    new_image,
    concatenate,
    blend_images,
    demo_basic_operations,
    demo_real_time_camera,
    demo_filters,
)

__version__ = "0.1.0"

# Logo path utilities
def get_logo_path(format='png'):
    """Get the path to the LiteCV logo in the specified format.

    Args:
        format: 'svg', 'png', or 'ascii'

    Returns:
        Path to the logo file
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', 'logo')
    if format == 'svg':
        return os.path.join(base_path, 'litecv_logo.svg')
    elif format == 'png':
        return os.path.join(base_path, 'logo.png')
    elif format == 'ascii':
        return os.path.join(base_path, 'litecv_ascii.txt')
    else:
        raise ValueError("Format must be 'svg', 'png', or 'ascii'")

def get_ascii_logo():
    """Get the ASCII art logo as a string.

    Returns:
        String containing the ASCII logo
    """
    try:
        with open(get_logo_path('ascii'), 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, UnicodeDecodeError):
        return """ _     _ _______ _______ _______
| |   | |  ____|  ____|  ____|
| |   | | |____| |____| |____
| |   | |_____ |_____ |_____ \
| |___|  _____| _____| _____| |
|_____| |_____| |_____| |_____|
       _______ _______ _______
      |  ____|  ____|  ____|
      | |____| |____| |____
      |_____ |_____ |_____ \
       _____| _____| _____| |
      |_____| |_____| |_____|
"""

__all__ = [
    "AdvancedLiteImage",
    "CameraFeed",
    "VideoProcessor",
    "StreamingFilter",
    "ObjectDetector",
    "RealTimeCameraApp",
    "FilterType",
    "open_image",
    "new_image",
    "concatenate",
    "blend_images",
    "demo_basic_operations",
    "demo_real_time_camera",
    "demo_filters",
    "get_logo_path",
    "get_ascii_logo",
    "__version__",
]
