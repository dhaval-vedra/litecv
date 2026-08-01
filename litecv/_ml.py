"""
litecv._ml — Optional ML inference bridge

Provides a thin, dependency-optional bridge for running ML models
on edge devices. Load a model once, run inference on LiteImage.

Supported backends (auto-detected):
  1. TFLite (tflite-runtime or tensorflow) — best for Pi/ARM
  2. ONNX Runtime (onnxruntime)
  3. PyTorch (torch) — if installed

Zero required dependencies — import succeeds even if no ML libs installed.

Usage:
    from litecv.ml import TFLiteModel, ONNXModel

    # TFLite (Raspberry Pi optimized)
    model = TFLiteModel("model.tflite")
    result = model.run(image)           # returns numpy array
    labels = model.classify(image)      # returns [(label, score)]

    # ONNX
    model = ONNXModel("model.onnx", input_shape=(1, 3, 224, 224))
    out = model.run(image)
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ._core import LiteImage


def _try_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter  # noqa
        return "tflite_runtime"
    except ImportError:
        pass
    try:
        import tensorflow as tf  # noqa
        return "tensorflow"
    except ImportError:
        pass
    return None


def _try_onnx():
    try:
        import onnxruntime  # noqa
        return True
    except ImportError:
        return False


class TFLiteModel:
    """
    TensorFlow Lite model runner — optimized for ARM/Raspberry Pi.

    Auto-selects between tflite_runtime (lightweight, Pi-optimised)
    and full tensorflow.

    Usage:
        model = TFLiteModel("efficientnet_lite0.tflite")
        labels = model.classify(image, top_k=5)
    """

    def __init__(self, model_path: str, num_threads: int = 4):
        backend = _try_tflite()
        if not backend:
            raise ImportError(
                "TFLite not available. Install: pip install tflite-runtime\n"
                "Raspberry Pi: pip install tflite-runtime --extra-index-url "
                "https://google-coral.github.io/py-repo/")

        if backend == "tflite_runtime":
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

        self._interp = Interpreter(model_path=model_path,
                                   num_threads=num_threads)
        self._interp.allocate_tensors()
        self._in_details  = self._interp.get_input_details()
        self._out_details = self._interp.get_output_details()
        self.input_shape  = tuple(self._in_details[0]["shape"])
        self.input_dtype  = self._in_details[0]["dtype"]

    def preprocess(self, img: LiteImage) -> np.ndarray:
        """Resize + normalize to model input shape."""
        _, h, w, _ = self.input_shape
        resized = img.to_rgb().resize(w, h)
        arr = resized._data.astype(self.input_dtype)
        if self.input_dtype == np.float32:
            arr = arr / 127.5 - 1.0  # standard MobileNet normalization
        return arr[np.newaxis]  # add batch dim

    def run(self, img: "LiteImage | np.ndarray") -> List[np.ndarray]:
        """Run inference. Returns list of output tensors."""
        if isinstance(img, LiteImage):
            tensor = self.preprocess(img)
        else:
            tensor = img
        self._interp.set_tensor(self._in_details[0]["index"], tensor)
        self._interp.invoke()
        return [self._interp.get_tensor(d["index"]) for d in self._out_details]

    def classify(self, img: LiteImage, top_k: int = 5,
                 labels: List[str] = None) -> List[Tuple[str, float]]:
        """
        Classification: returns top-k (label, score) pairs.
        Pass labels list or it returns indices as strings.
        """
        outputs = self.run(img)
        scores = outputs[0][0]
        if len(scores.shape) > 1:
            scores = scores.flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        result = []
        for idx in indices:
            label = labels[idx] if labels and idx < len(labels) else str(idx)
            result.append((label, float(scores[idx])))
        return result

    def detect(self, img: LiteImage,
               conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Object detection (SSD-style models).
        Returns list of dicts: {bbox, score, class_id}
        """
        outputs = self.run(img)
        # Standard SSD output format: boxes, classes, scores, count
        if len(outputs) >= 4:
            boxes   = outputs[0][0]   # [N, 4] — normalized [y1,x1,y2,x2]
            classes = outputs[1][0]   # [N]
            scores  = outputs[2][0]   # [N]
            count   = int(outputs[3][0])
        elif len(outputs) == 1:
            # Single output: just scores (classification)
            return self.classify(img)
        else:
            return []

        h, w = img.height, img.width
        result = []
        for i in range(count):
            if scores[i] < conf_threshold:
                continue
            y1, x1, y2, x2 = boxes[i]
            result.append({
                "x": int(x1 * w), "y": int(y1 * h),
                "w": int((x2-x1) * w), "h": int((y2-y1) * h),
                "score": float(scores[i]),
                "class_id": int(classes[i]),
            })
        return result

    @property
    def backend(self) -> str:
        return _try_tflite() or "unknown"


class ONNXModel:
    """
    ONNX Runtime model runner.

    Supports CPU and (if available) CUDA/TensorRT for NVIDIA Jetson.

    Usage:
        model = ONNXModel("yolov5s.onnx", input_shape=(1, 3, 640, 640))
        output = model.run(image)
    """

    def __init__(self, model_path: str,
                 input_shape: tuple = (1, 3, 224, 224),
                 providers: List[str] = None):
        if not _try_onnx():
            raise ImportError("onnxruntime not installed. pip install onnxruntime")
        import onnxruntime as ort
        self.input_shape = input_shape
        if providers is None:
            available = ort.get_available_providers()
            # Prefer GPU if available
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def preprocess(self, img: LiteImage) -> np.ndarray:
        """Resize + CHW format + normalize for standard vision models."""
        _, _, h, w = self.input_shape
        resized = img.to_rgb().resize(w, h)
        arr = resized._data.astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        return arr[np.newaxis]         # add batch dim

    def run(self, img: "LiteImage | np.ndarray") -> List[np.ndarray]:
        if isinstance(img, LiteImage):
            tensor = self.preprocess(img)
        else:
            tensor = img
        return self._session.run(self._output_names, {self._input_name: tensor})

    @property
    def providers(self) -> List[str]:
        return self._session.get_providers()


class NullModel:
    """
    Placeholder when no ML backend is available.
    Raises clear error on use — useful for optional-ML code paths.
    """

    def __init__(self, *args, **kwargs):
        self._backends = {
            "tflite": _try_tflite() is not None,
            "onnx":   _try_onnx(),
        }

    def run(self, *args, **kwargs):
        avail = [k for k, v in self._backends.items() if v]
        raise RuntimeError(
            f"No ML backend available. Install one of: tflite-runtime, onnxruntime\n"
            f"Available: {avail or 'none'}")

    @property
    def available_backends(self) -> Dict[str, bool]:
        return self._backends


def auto_model(model_path: str, **kwargs) -> "TFLiteModel | ONNXModel | NullModel":
    """
    Auto-select best model runner based on file extension and available backends.

    .tflite → TFLiteModel
    .onnx   → ONNXModel
    """
    if model_path.endswith(".tflite"):
        if _try_tflite():
            return TFLiteModel(model_path, **kwargs)
    elif model_path.endswith(".onnx"):
        if _try_onnx():
            return ONNXModel(model_path, **kwargs)
    return NullModel()
