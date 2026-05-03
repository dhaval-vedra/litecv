# LiteCV Logo Assets

This folder contains various logo assets for the LiteCV library.

## Files

- `logo.png` - **Main PNG logo** (primary logo used in README)
- `litecv_logo.svg` - Scalable vector logo (vector format, scalable)
- `litecv_logo_generated.png` - Generated PNG logo
- `litecv_ascii.txt` - ASCII art version for text-based contexts
- `generate_logo.py` - Python script to generate logo using LiteCV itself

## Usage

### In README.md
```markdown
<img src="logo/logo.png" alt="LiteCV Logo" width="200">
```

### In Python docstrings
```python
"""
LiteCV - Lightweight Computer Vision Library

 _     _ _______ _______ _______
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
```

### Generate PNG logo
```bash
python logo/generate_logo.py
```