cmake_minimum_required(VERSION 3.14)
project(lcv_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release build flags ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ARM NEON (Raspberry Pi)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
  message(STATUS "ARM detected — enabling NEON")
  add_compile_options(-mfpu=neon -mfloat-abi=hard)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-march=armv8-a+simd)
  else()
    add_compile_options(-march=armv7-a)
  endif()
endif()

# x86 SSE2 (Intel/AMD)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  message(STATUS "x86 detected — enabling SSE2")
  add_compile_options(-msse2)
  # Optionally enable AVX2 if supported
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(-mavx2)
    message(STATUS "AVX2 enabled")
  endif()
endif()

# General optimizations
add_compile_options(-O3 -ffast-math -funroll-loops)

# OpenMP (parallelism for batch operations)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found — parallel processing enabled")
endif()

# ── Find pybind11 ─────────────────────────────────────────────────────────────
execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT PYBIND11_DIR)
  execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

if(PYBIND11_DIR)
  list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
  find_package(pybind11 REQUIRED)
else()
  find_package(pybind11 REQUIRED)
endif()

# ── Extension module ──────────────────────────────────────────────────────────
pybind11_add_module(_lcv_ext MODULE
  lcv_bindings.cpp
)

target_include_directories(_lcv_ext PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/liteCV/core
  ${CMAKE_CURRENT_SOURCE_DIR}/liteCV/imgcodecs
  ${CMAKE_CURRENT_SOURCE_DIR}/liteCV/imgproc
)

# STB needs these defines in exactly one translation unit
target_compile_definitions(_lcv_ext PRIVATE
  STB_IMAGE_IMPLEMENTATION
  STB_IMAGE_WRITE_IMPLEMENTATION
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_lcv_ext PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Install: place extension next to litecv package ──────────────────────────
install(TARGETS _lcv_ext
  LIBRARY DESTINATION litecv
)
