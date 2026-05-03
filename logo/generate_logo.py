#!/usr/bin/env python3
"""
LiteCV Logo Generator
Creates a logo image using the LiteCV library itself.
"""

from litecv import new_image


def create_logo():
    """Create the LiteCV logo"""
    # Create a 200x80 image
    img = new_image(200, 80, color='white')

    # Draw "LiteCV" text
    img.draw_text('LiteCV', (20, 25), color='navy', size=24, font_path=None)

    # Draw a simple camera icon
    # Camera body
    img.draw_rectangle((140, 20), (180, 60), color='navy', width=2, fill='lightblue')

    # Lens
    img.draw_circle((160, 40), 12, color='navy', width=2, fill='white')

    # Lens center
    img.draw_circle((160, 40), 6, color='navy', width=1, fill='lightgray')

    # Save the logo
    img.save('logo/litecv_logo_generated.png')
    print('Logo saved as logo/litecv_logo_generated.png')


if __name__ == '__main__':
    create_logo()