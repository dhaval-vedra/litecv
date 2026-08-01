# Contributing to litecv-advanced 🚀

Welcome! We are thrilled that you are interested in contributing to **litecv-advanced**! Whether you are an open-source veteran or writing your very first line of code, we are excited to have you in our community.

This document will help you get set up and guide you through the contribution process.

---

## 🌟 Why Contribute?

- **Optimized for Edge:** Help build a high-performance computer vision library optimized for Raspberry Pi, Jetson Nano, and IoT devices.
- **Learn C++ & Python Bindings:** Work on real-world implementations of pybind11, SIMD vectorization, and Python-C++ hybrid architectures.
- **Join a Welcoming Community:** We value all types of contributions — from code optimization and feature requests to documentation improvements and tutorials.

---

## 🛠️ Developer Setup Guide

To contribute code, you'll want to set up your environment in development mode.

### 1. Fork and Clone the Repository
First, fork the repository on GitHub and clone your fork locally:
```bash
git clone https://github.com/your-username/litecv.git
cd litecv
```

### 2. Create a Virtual Environment
We recommend using a virtual environment (`venv`) to keep your dependencies isolated:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies
Install `litecv` in editable mode along with testing and package-building tools:
```bash
pip install -e .
pip install pybind11 pytest build wheel twine
```

### 4. Run the Test Suite
Ensure everything is working correctly by running the tests:
```bash
pytest tests/ -v
```
*You should see all unit tests pass!*

---

## 📂 Project Structure

Understanding where files live is key to making clean changes:

```
litecv/
├── 📦 litecv/            ← Pure Python API & Fallbacks
│   ├── __init__.py        ← Public API Exports
│   ├── _core.py           ← LiteImage core implementation
│   ├── _camera.py         ← Camera Feed (Sync/Async) backends
│   ├── _detection.py      ← Harris Corner, Optical Flow, Template Matching
│   ├── _pipeline.py       ← Declarative Pipeline Engine
│   └── _streaming.py      ← MJPEG HTTP Streaming Server
├── 📁 src/               ← C++ Accelerated Engine
│   ├── lcv_bindings.cpp   ← pybind11 module bindings
│   ├── lcv_simd.hpp       ← ARM NEON & x86 SSE/AVX auto-detect
│   ├── lcv_mempool.hpp    ← Thread-local bump allocator (no-malloc)
│   ├── lcv_advanced.hpp   ← CLAHE, Bilateral Filter, Harris Corners, HOG...
│   └── CMakeLists.txt     ← Build configuration for C++
├── 🧪 tests/             ← Comprehensive Pytest Suite
└── 📖 docs/              ← Detailed API & Usage Markdown Docs
```

---

## 🚀 How to Contribute

We follow a standard GitHub flow:

1. **Find/Open an Issue:** Look at our open issues or create a new one to discuss the feature or bug you want to work on.
2. **Create a Branch:** Create a branch with a descriptive name:
   ```bash
   git checkout -b feature/cool-new-filter
   ```
3. **Make Your Changes:**
   - Follow standard coding style (PEP 8 for Python).
   - Use type annotations.
   - Write clear docstrings.
4. **Add Tests:** If you are adding a new feature or fixing a bug, add a corresponding test in `tests/test_litecv_advanced.py`.
5. **Run Linting/Tests:** Ensure all tests pass (`pytest`).
6. **Submit a Pull Request (PR):** Push your branch to GitHub and open a Pull Request against our `main` branch. Provide a clear description of what your PR achieves.

---

## 💡 Great First Contribution Ideas

Unsure where to start? Here are some fantastic "good first issues" you can take on:

1. **Create a New Custom Filter 🎨**
   - Implement a new artistic filter (like custom duotone, posterization, or pixelation) in `_core.py` with a pure Python implementation and C++ optimization.
2. **Write a Jupyter Notebook Tutorial 📓**
   - Create a `tutorials/` directory and write a step-by-step notebook demonstrating object tracking, real-time streaming, or document scanning.
3. **Add More Unit Tests 🧪**
   - Increase coverage by writing tests for edge-cases, invalid inputs, or performance benchmarking.
4. **Improve the Documentation 📖**
   - Fix typos, improve API explanations, or translate README into different languages.

---

## 💬 Need Help?

If you have questions, get stuck, or want to discuss design decisions, please open an Issue or start a Discussion on GitHub. We're here to help you succeed!

Thank you for making **litecv-advanced** better! ❤️
