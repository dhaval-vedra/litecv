"""
setup.py for litecv-advanced

Builds the optional C++ extension (_lcv_ext) via CMake + pybind11.
If CMake or a compiler is not available, the build silently skips
the extension and the library falls back to pure Python (Pillow + NumPy).

Build:
    pip install .                          # auto C++ detection
    pip install . --no-build-isolation     # skip C++ (pure Python)
    CMAKE_BUILD_PARALLEL_LEVEL=4 pip install .  # parallel C++ build
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """Custom build backend that delegates to CMake."""

    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        src_dir  = Path(__file__).parent / "src"
        build_tmp = Path(self.build_temp) / ext.name

        # Check for prerequisites
        cmake = shutil.which("cmake")
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not cmake or not compiler:
            print(
                "\n[litecv] CMake or C++ compiler not found — "
                "building without C++ extension (pure-Python fallback).\n"
                "For best performance: sudo apt-get install cmake g++\n"
            )
            return

        # Check pybind11
        try:
            import pybind11  # noqa
        except ImportError:
            print("[litecv] pybind11 not found — building without C++ extension.\n"
                  "         pip install pybind11\n")
            return

        build_tmp.mkdir(parents=True, exist_ok=True)

        # CMake configure
        cmake_args = [
            cmake,
            str(src_dir),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        # Parallel jobs
        jobs = os.cpu_count() or 4
        build_args = ["--build", ".", "--config", "Release",
                      "--parallel", str(jobs)]

        try:
            subprocess.check_call(cmake_args, cwd=build_tmp)
            subprocess.check_call([cmake] + build_args, cwd=build_tmp)
        except subprocess.CalledProcessError as e:
            print(f"\n[litecv] C++ build failed ({e}) — falling back to pure Python.\n")

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)


setup(
    name="litecv-advanced",
    version="2.0.0",
    description="Lightweight, advanced computer vision for edge devices (Raspberry Pi, Jetson…)",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="litecv-advanced contributors",
    license="MIT",
    python_requires=">=3.8",
    packages=["litecv"],
    ext_modules=[Extension("_lcv_ext", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
    install_requires=[
        "Pillow>=9.0",
        "numpy>=1.21",
    ],
    extras_require={
        "ml":      ["tflite-runtime>=2.11"],
        "onnx":    ["onnxruntime>=1.14"],
        "camera":  ["picamera2"],
        "all":     ["tflite-runtime>=2.11", "onnxruntime>=1.14"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Embedded Systems",
    ],
    zip_safe=False,
)
