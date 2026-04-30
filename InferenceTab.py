import csv
import glob
import importlib
import os
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Detection:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]
    class_id: Optional[float] = None
    matched: bool = False
    match_iou: float = 0.0


@dataclass
class GroundTruth:
    label: str
    bbox: Tuple[float, float, float, float]
    matched: bool = False
    match_iou: float = 0.0


@dataclass
class Match:
    gt_index: int
    prediction_index: int
    iou: float


@dataclass
class ModelCandidate:
    path: str
    kind: str
    labels_path: Optional[str]


class DetectionModelError(Exception):
    pass


class BaseDetector:
    def __init__(self, model_path: str, labels: List[str], debug_logger=None):
        self.model_path = model_path
        self.labels = labels or ["object"]
        self.input_size = None
        self.debug_logger = debug_logger
        self.last_box_coordinate_mode = "unknown"

    def predict(
        self,
        image_bgr,
        score_threshold: float,
        debug: bool = False,
        image_name: str = ""
    ) -> List[Detection]:
        raise NotImplementedError

    def describe(self) -> List[str]:
        return []

    def _debug(self, message: str):
        if self.debug_logger:
            self.debug_logger(message)

    def _label_for_class(self, class_id) -> str:
        try:
            index = int(round(float(class_id)))
        except (TypeError, ValueError):
            return self.labels[0]

        for candidate in (index, index - 1):
            if 0 <= candidate < len(self.labels):
                return self.labels[candidate]
        return str(index)

    def _detections_from_output_items(
        self,
        output_items,
        image_shape,
        score_threshold: float,
        debug: bool = False,
        debug_prefix: str = ""
    ) -> List[Detection]:
        boxes, classes, scores, num_detections = self._extract_detection_tensors(output_items)
        if boxes is None or scores is None:
            if debug:
                self._debug(f"{debug_prefix} Could not map output tensors.")
                for name, value in output_items:
                    self._debug(f"{debug_prefix}   {self._array_summary(name, value)}")
            raise DetectionModelError("Could not identify detection boxes/scores in model outputs.")

        boxes = np.asarray(boxes)
        scores = np.asarray(scores).reshape(-1)
        if classes is None:
            classes = np.zeros_like(scores)
        else:
            classes = np.asarray(classes).reshape(-1)

        count = len(scores)
        if num_detections is not None:
            try:
                count = min(count, int(round(float(np.asarray(num_detections).reshape(-1)[0]))))
            except (TypeError, ValueError, IndexError):
                pass
        count = min(count, len(boxes), len(classes))

        image_h, image_w = image_shape[:2]
        converted_boxes = self._convert_boxes_to_original_image(boxes[:count], image_w, image_h)
        score_values = scores[:count]

        detections = []
        for index in range(count):
            score = float(scores[index])
            if score < score_threshold:
                continue
            class_id = classes[index]
            detections.append(
                Detection(
                    label=self._label_for_class(class_id),
                    score=score,
                    bbox=converted_boxes[index],
                    class_id=float(class_id)
                )
            )
        if debug:
            top_scores = sorted([float(score) for score in score_values], reverse=True)[:8]
            score_range = "none"
            if len(score_values):
                score_range = f"{float(np.nanmin(score_values)):.4f}..{float(np.nanmax(score_values)):.4f}"
            num_text = "none"
            if num_detections is not None:
                try:
                    num_text = str(float(np.asarray(num_detections).reshape(-1)[0]))
                except (TypeError, ValueError, IndexError):
                    num_text = "unreadable"
            self._debug(f"{debug_prefix} Output tensor summaries:")
            for name, value in output_items:
                self._debug(f"{debug_prefix}   {self._array_summary(name, value)}")
            self._debug(
                f"{debug_prefix} Parsed detections: raw_count={count}, "
                f"num_detections={num_text}, score_range={score_range}, "
                f"top_scores={[round(value, 4) for value in top_scores]}, "
                f"threshold={score_threshold}, kept={len(detections)}, "
                f"box_conversion={self.last_box_coordinate_mode}"
            )
        return detections

    def _convert_boxes_to_original_image(self, boxes, image_w: int, image_h: int):
        converted = []
        if boxes.size == 0:
            self.last_box_coordinate_mode = "none"
            return converted

        finite_boxes = boxes[np.isfinite(boxes)]
        max_value = float(np.max(finite_boxes)) if finite_boxes.size else 0.0
        input_w, input_h = self.input_size or (image_w, image_h)
        if max_value <= 1.5:
            self.last_box_coordinate_mode = "normalized -> original pixels"
        else:
            self.last_box_coordinate_mode = "input pixels -> original pixels"

        for box in boxes:
            y1, x1, y2, x2 = [float(value) for value in box[:4]]
            if max_value <= 1.5:
                xmin = x1 * image_w
                ymin = y1 * image_h
                xmax = x2 * image_w
                ymax = y2 * image_h
            else:
                xmin = x1 * image_w / max(1, input_w)
                ymin = y1 * image_h / max(1, input_h)
                xmax = x2 * image_w / max(1, input_w)
                ymax = y2 * image_h / max(1, input_h)

            xmin, xmax = sorted((max(0.0, xmin), min(float(image_w - 1), xmax)))
            ymin, ymax = sorted((max(0.0, ymin), min(float(image_h - 1), ymax)))
            converted.append((xmin, ymin, xmax, ymax))
        return converted

    @staticmethod
    def _array_summary(name, value) -> str:
        arr = np.asarray(value)
        summary = f"{name or '<unnamed>'}: shape={tuple(arr.shape)}, dtype={arr.dtype}"
        if arr.size and np.issubdtype(arr.dtype, np.number):
            finite = arr[np.isfinite(arr)]
            if finite.size:
                flat = finite.reshape(-1)
                sample = [float(value) for value in flat[:6]]
                summary += (
                    f", min={float(np.nanmin(finite)):.4f}, "
                    f"max={float(np.nanmax(finite)):.4f}, "
                    f"sample={[round(value, 4) for value in sample]}"
                )
        return summary

    @staticmethod
    def _squeeze_output(value):
        arr = np.asarray(value)
        while arr.ndim > 0 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        return arr

    @classmethod
    def _extract_detection_tensors(cls, output_items):
        arrays = []
        for name, value in output_items:
            arrays.append((str(name or "").lower(), cls._squeeze_output(value)))

        boxes = None
        classes = None
        scores = None
        num_detections = None

        for name, arr in arrays:
            if arr.ndim == 0 and ("num" in name or "count" in name):
                num_detections = arr
            elif arr.ndim >= 2 and arr.shape[-1] == 4 and ("box" in name or "location" in name):
                boxes = arr.reshape(-1, 4)
            elif arr.ndim >= 1 and ("score" in name or "confidence" in name):
                scores = arr.reshape(-1)
            elif arr.ndim >= 1 and ("class" in name or "category" in name or "label" in name):
                classes = arr.reshape(-1)

        if boxes is None:
            for _, arr in arrays:
                if arr.ndim >= 2 and arr.shape[-1] == 4:
                    boxes = arr.reshape(-1, 4)
                    break

        if num_detections is None:
            for _, arr in arrays:
                if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
                    value = float(arr.reshape(-1)[0])
                    if value >= 0:
                        num_detections = arr
                        break

        box_count = len(boxes) if boxes is not None else None
        one_dimensional = [
            arr.reshape(-1)
            for _, arr in arrays
            if arr.ndim >= 1 and (box_count is None or arr.reshape(-1).size == box_count)
        ]

        if scores is None:
            non_integer_like = []
            for arr in one_dimensional:
                if arr.size == 0:
                    continue
                if np.nanmax(arr) <= 1.01 and np.nanmin(arr) >= -0.01:
                    integer_like = np.nanmax(np.abs(arr - np.round(arr))) < 0.001
                    if not integer_like:
                        non_integer_like.append(arr)
            if non_integer_like:
                scores = non_integer_like[0]

        if classes is None:
            for arr in one_dimensional:
                if scores is not None and arr is scores:
                    continue
                if arr.size == 0:
                    continue
                integer_like = np.nanmax(np.abs(arr - np.round(arr))) < 0.001
                if integer_like:
                    classes = arr
                    break

        if scores is None and one_dimensional:
            for arr in one_dimensional:
                if classes is not None and arr is classes:
                    continue
                scores = arr
                break

        return boxes, classes, scores, num_detections


class TFLiteDetector(BaseDetector):
    def __init__(self, model_path: str, labels: List[str], debug_logger=None, model_threads: int = 1):
        super().__init__(model_path, labels, debug_logger=debug_logger)
        self.model_threads = max(1, int(model_threads or 1))
        self.interpreter = self._create_interpreter(model_path, self.model_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if not self.input_details:
            raise DetectionModelError("TFLite model has no input tensors.")

        shape = self.input_details[0].get("shape")
        if shape is None or len(shape) < 4:
            raise DetectionModelError("Expected TFLite input shape [1, height, width, 3].")
        if int(shape[3]) == 3:
            self.input_layout = "NHWC"
            self.input_h = int(shape[1])
            self.input_w = int(shape[2])
        elif int(shape[1]) == 3:
            self.input_layout = "NCHW"
            self.input_h = int(shape[2])
            self.input_w = int(shape[3])
        else:
            raise DetectionModelError(f"Expected a 3-channel TFLite input tensor; got shape {shape}.")
        self.input_size = (self.input_w, self.input_h)
        self.input_preprocess_description = "unknown"

    @staticmethod
    def _create_interpreter(model_path: str, model_threads: int):
        try:
            tensorflow = importlib.import_module("tensorflow")
            try:
                return tensorflow.lite.Interpreter(model_path=model_path, num_threads=model_threads)
            except TypeError:
                return tensorflow.lite.Interpreter(model_path=model_path)
        except Exception as tf_error:
            try:
                interpreter_module = importlib.import_module("tflite_runtime.interpreter")
                try:
                    return interpreter_module.Interpreter(model_path=model_path, num_threads=model_threads)
                except TypeError:
                    return interpreter_module.Interpreter(model_path=model_path)
            except Exception as runtime_error:
                raise DetectionModelError(
                    "Could not load TensorFlow Lite interpreter. "
                    f"TensorFlow error: {tf_error}; tflite_runtime error: {runtime_error}"
                )

    def describe(self) -> List[str]:
        lines = [
            f"TFLite raw interpreter: {self.model_path}",
            f"Input size: {self.input_w}x{self.input_h} layout={self.input_layout}",
            f"Requested TFLite interpreter threads: {self.model_threads}",
            "Runtime delegate: TensorFlow Lite Interpreter default CPU path "
            "(XNNPACK may be applied automatically by TensorFlow Lite; CUDA is not used here).",
        ]
        for detail in self.input_details:
            lines.append(
                f"Input tensor: name={detail.get('name')}, shape={detail.get('shape')}, "
                f"dtype={detail.get('dtype')}, quantization={detail.get('quantization')}"
            )
        for detail in self.output_details:
            lines.append(
                f"Output tensor: name={detail.get('name')}, shape={detail.get('shape')}, "
                f"dtype={detail.get('dtype')}, quantization={detail.get('quantization')}"
            )
        return lines

    def predict(
        self,
        image_bgr,
        score_threshold: float,
        debug: bool = False,
        image_name: str = ""
    ) -> List[Detection]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(resized, axis=0)
        if self.input_layout == "NCHW":
            input_data = np.transpose(input_data, (0, 3, 1, 2))

        input_detail = self.input_details[0]
        dtype = input_detail["dtype"]
        if np.issubdtype(dtype, np.floating):
            input_data = input_data.astype(dtype)
        else:
            info = np.iinfo(dtype)
            if dtype == np.uint8:
                input_data = np.clip(input_data, info.min, info.max).astype(dtype)
                self.input_preprocess_description = "RGB resize, direct uint8 image bytes"
            elif dtype == np.int8:
                scale, zero_point = input_detail.get("quantization", (0.0, 0))
                normalized = input_data.astype(np.float32) / 127.5 - 1.0
                if scale:
                    input_data = normalized / scale + zero_point
                    self.input_preprocess_description = "RGB resize, normalized [-1,1], quantized to int8"
                else:
                    input_data = input_data.astype(np.float32) - 128.0
                    self.input_preprocess_description = "RGB resize, shifted to int8"
                input_data = np.clip(input_data, info.min, info.max).astype(dtype)
            else:
                input_data = np.clip(input_data, info.min, info.max).astype(dtype)
                self.input_preprocess_description = f"RGB resize, direct cast to {dtype}"

        debug_prefix = f"[debug {image_name}]" if image_name else "[debug]"
        if debug:
            scale, zero_point = input_detail.get("quantization", (0.0, 0))
            quantized_note = ""
            if scale:
                real_min = (float(np.min(input_data)) - zero_point) * scale
                real_max = (float(np.max(input_data)) - zero_point) * scale
                quantized_note = f", dequantized_range={real_min:.3f}..{real_max:.3f}"
            self._debug(
                f"{debug_prefix} TFLite input: original={image_bgr.shape[1]}x{image_bgr.shape[0]}, "
                f"resized={self.input_w}x{self.input_h}, tensor_shape={tuple(input_data.shape)}, "
                f"dtype={input_data.dtype}, range={float(np.min(input_data)):.2f}..{float(np.max(input_data)):.2f}"
                f"{quantized_note}, preprocess={self.input_preprocess_description}"
            )

        self.interpreter.set_tensor(input_detail["index"], input_data)
        self.interpreter.invoke()
        output_items = [
            (detail.get("name", ""), self.interpreter.get_tensor(detail["index"]))
            for detail in self.output_details
        ]
        return self._detections_from_output_items(
            output_items,
            image_bgr.shape,
            score_threshold,
            debug=debug,
            debug_prefix=debug_prefix
        )


class TFLiteTaskDetector(BaseDetector):
    def __init__(self, model_path: str, labels: List[str], debug_logger=None, model_threads: int = 1):
        super().__init__(model_path, labels, debug_logger=debug_logger)
        self.model_threads = max(1, int(model_threads or 1))
        self.core, self.processor, self.vision = self._load_task_modules()
        try:
            base_options = self.core.BaseOptions(file_name=model_path, num_threads=self.model_threads)
        except TypeError:
            base_options = self.core.BaseOptions(file_name=model_path)
        detection_options = self.processor.DetectionOptions(max_results=100, score_threshold=0.0)
        options = self.vision.ObjectDetectorOptions(
            base_options=base_options,
            detection_options=detection_options
        )
        self.detector = self.vision.ObjectDetector.create_from_options(options)

    @staticmethod
    def _load_task_modules():
        import_errors = []
        module_sets = [
            ("tflite_support.task.core", "tflite_support.task.processor", "tflite_support.task.vision"),
            (
                "tensorflow_lite_support.python.task.core",
                "tensorflow_lite_support.python.task.processor",
                "tensorflow_lite_support.python.task.vision",
            ),
        ]
        for core_name, processor_name, vision_name in module_sets:
            try:
                return (
                    importlib.import_module(core_name),
                    importlib.import_module(processor_name),
                    importlib.import_module(vision_name),
                )
            except Exception as exc:
                import_errors.append(f"{vision_name}: {exc}")
        raise DetectionModelError("Could not load TFLite Support ObjectDetector: " + " | ".join(import_errors))

    def describe(self) -> List[str]:
        return [
            f"TFLite Support ObjectDetector: {self.model_path}",
            f"Requested TFLite interpreter threads: {self.model_threads}",
            "Input preprocessing and output decoding are handled by the TFLite Task Library.",
            "Runtime delegate: TFLite Task Library default CPU path "
            "(delegate choice is runtime-specific; CUDA is not used by this desktop path).",
        ]

    def predict(
        self,
        image_bgr,
        score_threshold: float,
        debug: bool = False,
        image_name: str = ""
    ) -> List[Detection]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor_image = self.vision.TensorImage.create_from_array(image_rgb)
        detection_result = self.detector.detect(tensor_image)
        detections = []

        raw_detections = list(getattr(detection_result, "detections", []) or [])
        for detection in raw_detections:
            categories = list(getattr(detection, "categories", []) or [])
            if not categories:
                continue
            category = max(categories, key=lambda item: float(getattr(item, "score", 0.0)))
            score = float(getattr(category, "score", 0.0))
            if score < score_threshold:
                continue

            bbox = getattr(detection, "bounding_box", None)
            if bbox is None:
                continue
            origin_x = float(getattr(bbox, "origin_x", 0.0))
            origin_y = float(getattr(bbox, "origin_y", 0.0))
            width = float(getattr(bbox, "width", 0.0))
            height = float(getattr(bbox, "height", 0.0))
            class_id = getattr(category, "index", None)
            label = str(getattr(category, "category_name", "") or "").strip()
            if not label or set(label) == {"?"}:
                label = self._label_for_class(class_id)

            detections.append(
                Detection(
                    label=label,
                    score=score,
                    bbox=(origin_x, origin_y, origin_x + width, origin_y + height),
                    class_id=float(class_id) if class_id is not None else None,
                )
            )

        if debug:
            debug_prefix = f"[debug {image_name}]" if image_name else "[debug]"
            scores = []
            for detection in raw_detections:
                for category in getattr(detection, "categories", []) or []:
                    scores.append(float(getattr(category, "score", 0.0)))
            top_scores = sorted(scores, reverse=True)[:8]
            self._debug(
                f"{debug_prefix} TFLite Support detector: image={image_bgr.shape[1]}x{image_bgr.shape[0]}, "
                f"raw_detections={len(raw_detections)}, "
                f"top_scores={[round(value, 4) for value in top_scores]}, "
                f"threshold={score_threshold}, kept={len(detections)}"
            )

        return detections


class SavedModelDetector(BaseDetector):
    def __init__(self, model_path: str, labels: List[str], debug_logger=None):
        super().__init__(model_path, labels, debug_logger=debug_logger)
        try:
            self.tensorflow = importlib.import_module("tensorflow")
        except Exception as exc:
            raise DetectionModelError(f"TensorFlow is required to load SavedModel folders: {exc}")

        self.model = self.tensorflow.saved_model.load(model_path)
        signatures = getattr(self.model, "signatures", {})
        self.signature = signatures.get("serving_default")
        if self.signature is None and signatures:
            self.signature = next(iter(signatures.values()))
        if self.signature is None:
            raise DetectionModelError("SavedModel does not expose a serving signature.")

        self.input_name, self.input_spec = self._first_input_spec()
        self.input_shape = self._shape_as_list(getattr(self.input_spec, "shape", None))
        self.input_dtype = getattr(self.input_spec, "dtype", None)

        target_h = self._fixed_dim(self.input_shape, 1)
        target_w = self._fixed_dim(self.input_shape, 2)
        if target_h and target_w:
            self.input_size = (target_w, target_h)

    def describe(self) -> List[str]:
        input_shape = getattr(self.input_spec, "shape", None)
        return [
            f"SavedModel: {self.model_path}",
            f"Serving input: name={self.input_name}, shape={input_shape}, dtype={self.input_dtype}",
            f"Signature outputs: {', '.join(self.signature.structured_outputs.keys())}",
        ]

    def _first_input_spec(self):
        structured = self.signature.structured_input_signature
        positional_args, keyword_args = structured
        if keyword_args:
            name, spec = next(iter(keyword_args.items()))
            return name, spec
        if positional_args:
            return None, positional_args[0]
        raise DetectionModelError("SavedModel serving signature has no inputs.")

    @staticmethod
    def _shape_as_list(shape):
        if shape is None:
            return []
        try:
            return shape.as_list()
        except AttributeError:
            return list(shape)

    @staticmethod
    def _fixed_dim(shape, index):
        if not shape or len(shape) <= index:
            return None
        value = shape[index]
        if value in (None, -1):
            return None
        return int(value)

    def predict(
        self,
        image_bgr,
        score_threshold: float,
        debug: bool = False,
        image_name: str = ""
    ) -> List[Detection]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target_h = self._fixed_dim(self.input_shape, 1)
        target_w = self._fixed_dim(self.input_shape, 2)
        if target_h and target_w:
            model_image = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            self.input_size = (target_w, target_h)
        else:
            model_image = image_rgb
            self.input_size = (image_bgr.shape[1], image_bgr.shape[0])

        input_data = np.expand_dims(model_image, axis=0)
        dtype_name = getattr(self.input_dtype, "name", "")
        if "float" in dtype_name:
            input_data = input_data.astype(np.float32)
        elif "int" in dtype_name:
            input_data = input_data.astype(np.int32 if dtype_name == "int32" else np.uint8)

        debug_prefix = f"[debug {image_name}]" if image_name else "[debug]"
        if debug:
            self._debug(
                f"{debug_prefix} SavedModel input: original={image_bgr.shape[1]}x{image_bgr.shape[0]}, "
                f"model_image={model_image.shape[1]}x{model_image.shape[0]}, "
                f"tensor_shape={tuple(input_data.shape)}, dtype={input_data.dtype}, "
                f"range={float(np.min(input_data)):.2f}..{float(np.max(input_data)):.2f}"
            )

        tensor = self.tensorflow.convert_to_tensor(input_data, dtype=self.input_dtype)
        if self.input_name:
            outputs = self.signature(**{self.input_name: tensor})
        else:
            outputs = self.signature(tensor)

        output_items = []
        for name, value in outputs.items():
            if hasattr(value, "numpy"):
                value = value.numpy()
            output_items.append((name, value))

        return self._detections_from_output_items(
            output_items,
            image_bgr.shape,
            score_threshold,
            debug=debug,
            debug_prefix=debug_prefix
        )


class InferenceTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.results = []
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.preview_window = None
        self.preview_image_label = None
        self.preview_details_text = None
        self.preview_photo = None
        self.preview_result_index = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        title = ttk.Label(main_frame, text="Inference Review", font=("TkDefaultFont", 12, "bold"))
        title.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        model_frame.columnconfigure(1, weight=1)

        self.model_path_var = tk.StringVar()
        ttk.Label(model_frame, text="Saved model/run folder or .tflite:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=70).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(model_frame, text="Browse Folder", command=self.browse_model_folder).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(model_frame, text="Browse TFLite", command=self.browse_model_file).grid(row=0, column=3, padx=(6, 0))

        self.model_info_var = tk.StringVar(value="Select a training run folder, saved_model folder, or .tflite file.")
        ttk.Label(model_frame, textvariable=self.model_info_var).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

        data_frame = ttk.LabelFrame(main_frame, text="Dataset", padding="10")
        data_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        data_frame.columnconfigure(1, weight=1)

        self.dataset_dir_var = tk.StringVar()
        ttk.Label(data_frame, text="Image/XML directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Entry(data_frame, textvariable=self.dataset_dir_var, width=70).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(data_frame, text="Browse", command=self.browse_dataset_dir).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(data_frame, text="Use db_final_data", command=self.use_default_dataset_root).grid(row=0, column=3, padx=(6, 0))

        options_frame = ttk.LabelFrame(main_frame, text="Review Options", padding="10")
        options_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self.score_threshold_var = tk.StringVar(value="0.50")
        self.iou_threshold_var = tk.StringVar(value="0.50")
        self.max_images_var = tk.StringVar()
        self.debug_images_var = tk.StringVar(value="5")
        default_workers = max(1, min(4, os.cpu_count() or 4))
        self.parallel_workers_var = tk.StringVar(value=str(default_workers))
        self.model_threads_var = tk.StringVar(value="1")
        self.recursive_var = tk.BooleanVar(value=False)
        self.failures_only_var = tk.BooleanVar(value=True)

        ttk.Label(options_frame, text="Score threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(options_frame, textvariable=self.score_threshold_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=(0, 14))
        ttk.Label(options_frame, text="IoU match threshold:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Entry(options_frame, textvariable=self.iou_threshold_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=(0, 14))
        ttk.Label(options_frame, text="Max images:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        ttk.Entry(options_frame, textvariable=self.max_images_var, width=8).grid(row=0, column=5, sticky=tk.W, padx=(0, 14))
        ttk.Label(options_frame, text="Debug first:").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        ttk.Entry(options_frame, textvariable=self.debug_images_var, width=8).grid(row=0, column=7, sticky=tk.W, padx=(0, 14))
        ttk.Label(options_frame, text="Workers:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(8, 0))
        ttk.Entry(options_frame, textvariable=self.parallel_workers_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=(0, 14), pady=(8, 0))
        ttk.Label(options_frame, text="Threads/model:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(8, 0))
        ttk.Entry(options_frame, textvariable=self.model_threads_var, width=8).grid(row=1, column=3, sticky=tk.W, padx=(0, 14), pady=(8, 0))
        ttk.Checkbutton(options_frame, text="Recursive", variable=self.recursive_var).grid(row=1, column=4, sticky=tk.W, padx=(0, 14), pady=(8, 0))
        ttk.Checkbutton(
            options_frame,
            text="Show failures only",
            variable=self.failures_only_var,
            command=self.refresh_results_tree
        ).grid(row=1, column=5, sticky=tk.W, pady=(8, 0))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, sticky=tk.W, pady=(0, 8))

        self.run_button = ttk.Button(button_frame, text="Run Inference Review", command=self.run_inference)
        self.run_button.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_inference, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="Preview Selected", command=self.preview_selected_result).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="Export CSV", command=self.export_results_csv).pack(side=tk.LEFT, padx=6)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var, width=32).grid(row=0, column=1, sticky=tk.E, padx=(8, 0))

        summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding="8")
        summary_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        summary_frame.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_frame, height=7, wrap=tk.WORD)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        results_frame = ttk.LabelFrame(main_frame, text="Image Results", padding="8")
        results_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        results_frame.columnconfigure(0, weight=1)

        columns = (
            "index", "status", "file", "gt", "pred", "matches", "missed",
            "false_pos", "best_iou", "top_prediction", "notes"
        )
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=14)
        headings = {
            "index": "#",
            "status": "Status",
            "file": "Image",
            "gt": "GT",
            "pred": "Pred",
            "matches": "Match",
            "missed": "Miss",
            "false_pos": "FP",
            "best_iou": "Best IoU",
            "top_prediction": "Top Prediction",
            "notes": "Notes",
        }
        widths = {
            "index": 45,
            "status": 90,
            "file": 260,
            "gt": 45,
            "pred": 45,
            "matches": 55,
            "missed": 55,
            "false_pos": 55,
            "best_iou": 70,
            "top_prediction": 180,
            "notes": 360,
        }
        for column in columns:
            self.results_tree.heading(column, text=headings[column])
            self.results_tree.column(column, width=widths[column], anchor=tk.W)
        self.results_tree.column("index", anchor=tk.CENTER, stretch=False)
        self.results_tree.column("gt", anchor=tk.CENTER, stretch=False)
        self.results_tree.column("pred", anchor=tk.CENTER, stretch=False)
        self.results_tree.column("matches", anchor=tk.CENTER, stretch=False)
        self.results_tree.column("missed", anchor=tk.CENTER, stretch=False)
        self.results_tree.column("false_pos", anchor=tk.CENTER, stretch=False)
        self.results_tree.column("best_iou", anchor=tk.CENTER, stretch=False)

        y_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        x_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E))
        y_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        x_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.results_tree.bind("<Double-Button-1>", lambda event: self.preview_selected_result())

        self._write_summary("Run inference to compare model predictions against paired XML annotations.")

    def browse_model_folder(self):
        path = filedialog.askdirectory(title="Select model run or saved_model folder", initialdir=self.app_root)
        if path:
            self.model_path_var.set(path)
            self.update_model_info()

    def browse_model_file(self):
        path = filedialog.askopenfilename(
            title="Select TFLite model",
            initialdir=self.app_root,
            filetypes=[("TFLite model", "*.tflite"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)
            self.update_model_info()

    def browse_dataset_dir(self):
        path = filedialog.askdirectory(title="Select image/XML dataset directory", initialdir=self.app_root)
        if path:
            self.dataset_dir_var.set(path)

    def use_default_dataset_root(self):
        default_path = os.path.join(self.app_root, "db_final_data")
        self.dataset_dir_var.set(default_path if os.path.exists(default_path) else self.app_root)
        self.recursive_var.set(True)

    def update_model_info(self):
        try:
            candidate = self.resolve_model_candidate(self.model_path_var.get())
            labels = self.read_labels(candidate.labels_path)
            labels_text = ", ".join(labels[:6]) if labels else "object"
            self.model_info_var.set(f"Will load {candidate.kind}: {candidate.path} | labels: {labels_text}")
        except Exception as exc:
            self.model_info_var.set(str(exc))

    def run_inference(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Info", "Inference is already running.")
            return

        model_path = self.model_path_var.get().strip()
        dataset_dir = self.dataset_dir_var.get().strip()
        if not model_path:
            messagebox.showwarning("Warning", "Select a model folder or .tflite file first.")
            return
        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showwarning("Warning", "Select a dataset directory first.")
            return

        try:
            score_threshold = self._parse_float(self.score_threshold_var.get(), "Score threshold", 0.0, 1.0)
            iou_threshold = self._parse_float(self.iou_threshold_var.get(), "IoU threshold", 0.0, 1.0)
            max_images = self._parse_optional_int(self.max_images_var.get(), "Max images")
            debug_images = self._parse_optional_int(self.debug_images_var.get(), "Debug first", allow_zero=True)
            parallel_workers = self._parse_optional_int(self.parallel_workers_var.get(), "Workers")
            model_threads = self._parse_optional_int(self.model_threads_var.get(), "Threads/model")
        except ValueError as exc:
            messagebox.showwarning("Invalid Options", str(exc))
            return

        self.results = []
        self.refresh_results_tree()
        self._write_summary("Loading model...")
        self.progress_var.set(0)
        self.status_var.set("Loading model")
        self.stop_event.clear()
        self._set_running(True)

        args = (
            model_path,
            dataset_dir,
            score_threshold,
            iou_threshold,
            max_images,
            self.recursive_var.get(),
            debug_images if debug_images is not None else 0,
            parallel_workers or 1,
            model_threads or 1,
        )
        self.worker_thread = threading.Thread(target=self._run_inference_thread, args=args, daemon=True)
        self.worker_thread.start()

    def stop_inference(self):
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def _set_running(self, running: bool):
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    @staticmethod
    def _parse_float(raw_value: str, name: str, minimum: float, maximum: float) -> float:
        try:
            value = float(str(raw_value).strip())
        except ValueError:
            raise ValueError(f"{name} must be a number.")
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        return value

    @staticmethod
    def _parse_optional_int(raw_value: str, name: str, allow_zero: bool = False):
        raw_value = str(raw_value).strip()
        if not raw_value:
            return None
        try:
            value = int(raw_value)
        except ValueError:
            raise ValueError(f"{name} must be a whole number.")
        if allow_zero:
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        elif value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return value

    def _run_inference_thread(
        self,
        model_path,
        dataset_dir,
        score_threshold,
        iou_threshold,
        max_images,
        recursive,
        debug_images,
        parallel_workers,
        model_threads
    ):
        started = time.time()
        completed_results = []
        stopped = False

        try:
            candidate = self.resolve_model_candidate(model_path)
            labels = self.read_labels(candidate.labels_path)
            self._log(f"\n=== Inference Review ===")
            self._log(f"Model: {candidate.path}")
            self._log(f"Model type: {candidate.kind}")
            self._log(f"Labels: {', '.join(labels)}")
            self._log(f"Dataset: {dataset_dir}")
            self._log(f"Score threshold: {score_threshold}")
            self._log(f"IoU threshold: {iou_threshold}")
            self._log(f"Debug logging: first {debug_images} image(s)")
            self._log(f"Parallel workers: {parallel_workers}")
            self._log(f"TFLite threads per model instance: {model_threads}")

            image_paths = self.find_image_files(dataset_dir, recursive)
            if max_images:
                image_paths = image_paths[:max_images]
            if not image_paths:
                self._after(self._run_finished, [], 0.0, False, "No images found.")
                return

            total = len(image_paths)
            parallel_workers = max(1, min(int(parallel_workers or 1), total))
            completed_results = [None] * total
            completed_count = 0

            thread_local = threading.local()
            describe_logged = False
            describe_lock = threading.Lock()

            def detector_for_worker():
                nonlocal describe_logged
                detector = getattr(thread_local, "detector", None)
                if detector is None:
                    worker_name = threading.current_thread().name
                    detector = self.create_detector(
                        candidate,
                        labels,
                        model_threads=model_threads,
                        worker_name=worker_name
                    )
                    thread_local.detector = detector
                    self._log(f"{worker_name}: loaded model instance")
                    with describe_lock:
                        if not describe_logged:
                            for line in detector.describe():
                                self._log(line)
                            describe_logged = True
                return detector

            def evaluate_indexed_image(index, image_path):
                if self.stop_event.is_set():
                    return None
                detector = detector_for_worker()
                result = self.evaluate_image(
                    image_path,
                    detector,
                    score_threshold,
                    iou_threshold,
                    debug=index < debug_images
                )
                result["index"] = index
                return result

            executor = ThreadPoolExecutor(
                max_workers=parallel_workers,
                thread_name_prefix="InferenceWorker"
            )
            futures = [
                executor.submit(evaluate_indexed_image, index, image_path)
                for index, image_path in enumerate(image_paths)
            ]

            try:
                for future in as_completed(futures):
                    if self.stop_event.is_set():
                        stopped = True
                        break

                    result = future.result()
                    if result is None:
                        stopped = True
                        continue

                    completed_results[result["index"]] = result
                    completed_count += 1
                    self._after(self._add_result, result)
                    self._after(
                        self._update_progress,
                        completed_count,
                        total,
                        Path(result["image_path"]).name
                    )
            finally:
                if stopped:
                    for future in futures:
                        future.cancel()
                executor.shutdown(wait=True, cancel_futures=stopped)

            elapsed = time.time() - started
            ordered_results = [result for result in completed_results if result is not None]
            self._after(self._run_finished, ordered_results, elapsed, stopped, None)
        except Exception as exc:
            elapsed = time.time() - started
            ordered_results = [result for result in completed_results if result is not None]
            self._after(self._run_finished, ordered_results, elapsed, stopped, str(exc))

    def create_detector(
        self,
        candidate: ModelCandidate,
        labels: List[str],
        model_threads: int = 1,
        worker_name: str = ""
    ):
        prefix = f"{worker_name}: " if worker_name else ""
        if candidate.kind == "tflite":
            try:
                detector = TFLiteTaskDetector(
                    candidate.path,
                    labels,
                    debug_logger=self._log,
                    model_threads=model_threads
                )
                self._log(f"{prefix}Using TFLite Support ObjectDetector for metadata-aware preprocessing/postprocessing.")
                return detector
            except Exception as exc:
                self._log(f"{prefix}TFLite Support ObjectDetector unavailable; falling back to raw interpreter: {exc}")
                return TFLiteDetector(
                    candidate.path,
                    labels,
                    debug_logger=self._log,
                    model_threads=model_threads
                )
        return SavedModelDetector(candidate.path, labels, debug_logger=self._log)

    def evaluate_image(
        self,
        image_path: str,
        detector,
        score_threshold: float,
        iou_threshold: float,
        debug: bool = False
    ) -> Dict:
        image = cv2.imread(image_path)
        xml_path = self.xml_path_for_image(image_path)
        notes = []

        if image is None:
            return self.build_error_result(image_path, xml_path, "Could not read image.")

        gt_boxes = []
        xml_found = os.path.exists(xml_path)
        if xml_found:
            try:
                gt_boxes = self.parse_pascal_voc_xml(xml_path)
            except Exception as exc:
                notes.append(f"XML parse error: {exc}")
        else:
            notes.append("No paired XML")

        try:
            predictions = detector.predict(
                image,
                score_threshold,
                debug=debug,
                image_name=Path(image_path).name
            )
        except Exception as exc:
            return self.build_error_result(image_path, xml_path, f"Inference error: {exc}", gt_boxes)

        if not predictions:
            notes.append(f"No predictions above score threshold {score_threshold:.2f}")

        comparison = self.compare_detections(gt_boxes, predictions, iou_threshold)
        status = self.status_for_result(xml_found, gt_boxes, predictions, comparison, notes)
        best_iou = comparison["best_iou"]
        top_prediction = self.format_top_prediction(predictions)
        notes.extend(self.notes_for_result(comparison))

        return {
            "image_path": image_path,
            "xml_path": xml_path,
            "xml_found": xml_found,
            "status": status,
            "gt": gt_boxes,
            "predictions": predictions,
            "matches": comparison["matches"],
            "missed": comparison["missed"],
            "false_positives": comparison["false_positives"],
            "label_mismatches": comparison["label_mismatches"],
            "best_iou": best_iou,
            "top_prediction": top_prediction,
            "notes": "; ".join(notes),
        }

    @staticmethod
    def build_error_result(image_path: str, xml_path: str, error: str, gt_boxes=None) -> Dict:
        return {
            "image_path": image_path,
            "xml_path": xml_path,
            "xml_found": os.path.exists(xml_path),
            "status": "ERROR",
            "gt": gt_boxes or [],
            "predictions": [],
            "matches": [],
            "missed": list(range(len(gt_boxes or []))),
            "false_positives": [],
            "label_mismatches": [],
            "best_iou": 0.0,
            "top_prediction": "",
            "notes": error,
        }

    @staticmethod
    def xml_path_for_image(image_path: str) -> str:
        return str(Path(image_path).with_suffix(".xml"))

    @classmethod
    def parse_pascal_voc_xml(cls, xml_path: str) -> List[GroundTruth]:
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "object").strip() or "object"
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = cls._parse_number(bbox.findtext("xmin"))
            ymin = cls._parse_number(bbox.findtext("ymin"))
            xmax = cls._parse_number(bbox.findtext("xmax"))
            ymax = cls._parse_number(bbox.findtext("ymax"))
            if None in (xmin, ymin, xmax, ymax):
                continue
            xmin, xmax = sorted((xmin, xmax))
            ymin, ymax = sorted((ymin, ymax))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append(GroundTruth(label=label, bbox=(xmin, ymin, xmax, ymax)))
        return boxes

    @staticmethod
    def _parse_number(value):
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def normalize_label(label: str) -> str:
        return str(label or "").strip().lower()

    @classmethod
    def compare_detections(cls, ground_truths: List[GroundTruth], predictions: List[Detection], iou_threshold: float):
        for gt in ground_truths:
            gt.matched = False
            gt.match_iou = 0.0
        for prediction in predictions:
            prediction.matched = False
            prediction.match_iou = 0.0

        candidates = []
        best_iou = 0.0
        label_mismatches = []

        for gt_index, gt in enumerate(ground_truths):
            for pred_index, prediction in enumerate(predictions):
                iou = cls.calculate_iou(gt.bbox, prediction.bbox)
                best_iou = max(best_iou, iou)
                same_label = cls.normalize_label(gt.label) == cls.normalize_label(prediction.label)
                if same_label and iou >= iou_threshold:
                    candidates.append((iou, gt_index, pred_index))
                elif not same_label and iou >= iou_threshold:
                    label_mismatches.append((gt_index, pred_index, iou))

        matches = []
        used_gt = set()
        used_predictions = set()
        for iou, gt_index, pred_index in sorted(candidates, reverse=True):
            if gt_index in used_gt or pred_index in used_predictions:
                continue
            used_gt.add(gt_index)
            used_predictions.add(pred_index)
            ground_truths[gt_index].matched = True
            ground_truths[gt_index].match_iou = iou
            predictions[pred_index].matched = True
            predictions[pred_index].match_iou = iou
            matches.append(Match(gt_index=gt_index, prediction_index=pred_index, iou=iou))

        missed = [index for index in range(len(ground_truths)) if index not in used_gt]
        false_positives = [index for index in range(len(predictions)) if index not in used_predictions]

        return {
            "matches": matches,
            "missed": missed,
            "false_positives": false_positives,
            "label_mismatches": label_mismatches,
            "best_iou": best_iou,
        }

    @staticmethod
    def calculate_iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        intersection = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def status_for_result(xml_found, ground_truths, predictions, comparison, notes) -> str:
        if notes and any(note.startswith("XML parse error") for note in notes):
            return "XML_ERROR"
        if not xml_found:
            return "NO_XML"
        if not ground_truths and not predictions:
            return "EMPTY"
        if not comparison["missed"] and not comparison["false_positives"]:
            return "PASS"
        if comparison["missed"] and comparison["false_positives"]:
            return "MIXED"
        if comparison["missed"]:
            return "MISSED"
        return "FALSE_POS"

    @staticmethod
    def notes_for_result(comparison) -> List[str]:
        notes = []
        if comparison["label_mismatches"]:
            notes.append(f"{len(comparison['label_mismatches'])} label mismatch overlap(s)")
        return notes

    @staticmethod
    def format_top_prediction(predictions: List[Detection]) -> str:
        if not predictions:
            return ""
        top = max(predictions, key=lambda item: item.score)
        return f"{top.label} {top.score:.2f}"

    @staticmethod
    def find_image_files(dataset_dir: str, recursive: bool) -> List[str]:
        pattern = "**/*" if recursive else "*"
        files = []
        for path in glob.glob(os.path.join(dataset_dir, pattern), recursive=recursive):
            if os.path.isfile(path) and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                files.append(path)
        return sorted(files)

    @staticmethod
    def resolve_model_candidate(raw_path: str) -> ModelCandidate:
        path = os.path.abspath(os.path.expanduser(str(raw_path).strip()))
        if not path:
            raise DetectionModelError("No model path selected.")
        if os.path.isfile(path):
            if Path(path).suffix.lower() != ".tflite":
                raise DetectionModelError("Model file must be a .tflite file.")
            return ModelCandidate(path=path, kind="tflite", labels_path=InferenceTab.find_labels_near(path))
        if not os.path.isdir(path):
            raise DetectionModelError(f"Model path does not exist: {path}")

        if os.path.exists(os.path.join(path, "saved_model.pb")):
            return ModelCandidate(path=path, kind="saved_model", labels_path=InferenceTab.find_labels_near(path))

        tflite_files = sorted(
            glob.glob(os.path.join(path, "*.tflite")),
            key=lambda item: os.path.getmtime(item),
            reverse=True
        )
        if tflite_files:
            return ModelCandidate(
                path=tflite_files[0],
                kind="tflite",
                labels_path=InferenceTab.find_labels_near(tflite_files[0])
            )

        nested_saved_model = os.path.join(path, "saved_model")
        if os.path.exists(os.path.join(nested_saved_model, "saved_model.pb")):
            return ModelCandidate(
                path=nested_saved_model,
                kind="saved_model",
                labels_path=InferenceTab.find_labels_near(nested_saved_model)
            )

        raise DetectionModelError("Could not find saved_model.pb or a .tflite file in the selected folder.")

    @staticmethod
    def find_labels_near(model_path: str) -> Optional[str]:
        candidates = []
        if os.path.isfile(model_path):
            candidates.append(os.path.join(os.path.dirname(model_path), "labels.txt"))
        else:
            candidates.append(os.path.join(model_path, "labels.txt"))
            candidates.append(os.path.join(os.path.dirname(model_path), "labels.txt"))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def read_labels(labels_path: Optional[str]) -> List[str]:
        if not labels_path or not os.path.exists(labels_path):
            return ["object"]
        labels = []
        with open(labels_path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                label = line.strip()
                if not label:
                    continue
                if set(label) == {"?"}:
                    continue
                if label not in labels:
                    labels.append(label)
        return labels or ["object"]

    def _add_result(self, result):
        index = result["index"]
        while len(self.results) <= index:
            self.results.append(None)
        self.results[index] = result
        if self.failures_only_var.get() and not self.is_failure(result):
            return
        item_id = str(index)
        if self.results_tree.exists(item_id):
            self.results_tree.delete(item_id)
        self.insert_result_row(result)

    def insert_result_row(self, result):
        self.results_tree.insert("", tk.END, iid=str(result["index"]), values=self.row_values(result))

    def row_values(self, result):
        return (
            result["index"] + 1,
            result["status"],
            Path(result["image_path"]).name,
            len(result["gt"]),
            len(result["predictions"]),
            len(result["matches"]),
            len(result["missed"]),
            len(result["false_positives"]),
            f"{result['best_iou']:.2f}",
            result["top_prediction"],
            result["notes"],
        )

    def refresh_results_tree(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for result in self.results:
            if result is None:
                continue
            if self.failures_only_var.get() and not self.is_failure(result):
                continue
            self.insert_result_row(result)

    @staticmethod
    def is_failure(result) -> bool:
        return result["status"] not in {"PASS", "EMPTY"}

    def _update_progress(self, current, total, filename):
        value = (current / total) * 100 if total else 0
        self.progress_var.set(value)
        self.status_var.set(f"{current}/{total}: {filename}")

    def _run_finished(self, results, elapsed, stopped, error):
        self._set_running(False)
        if error:
            self.status_var.set("Error")
            self._log(f"Inference review failed: {error}")
            messagebox.showerror("Inference Review Failed", error)
        elif stopped:
            self.status_var.set("Stopped")
            self._log("Inference review stopped by user.")
        else:
            self.progress_var.set(100)
            self.status_var.set("Complete")
            self._log(f"Inference review complete: {len(results)} image(s) in {elapsed:.1f}s")
        self._write_summary(self.build_summary(results, elapsed, stopped, error))

    def build_summary(self, results, elapsed=0.0, stopped=False, error=None) -> str:
        results = [result for result in results if result is not None]
        if error:
            return f"Run failed: {error}"
        if not results:
            return "No results yet."

        total = len(results)
        failures = [result for result in results if self.is_failure(result)]
        matches = sum(len(result["matches"]) for result in results)
        missed = sum(len(result["missed"]) for result in results)
        false_positives = sum(len(result["false_positives"]) for result in results)
        gt_total = sum(len(result["gt"]) for result in results)
        pred_total = sum(len(result["predictions"]) for result in results)
        precision_denominator = matches + false_positives
        recall_denominator = matches + missed
        precision = matches / precision_denominator if precision_denominator else 0.0
        recall = matches / recall_denominator if recall_denominator else 0.0

        status_counts = {}
        for result in results:
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        status_text = ", ".join(f"{key}: {value}" for key, value in sorted(status_counts.items()))

        prefix = "Stopped. " if stopped else ""
        return (
            f"{prefix}Reviewed {total} image(s) in {elapsed:.1f}s.\n"
            f"Failures needing review: {len(failures)} | Status counts: {status_text}\n"
            f"Ground truth boxes: {gt_total} | Predictions above threshold: {pred_total}\n"
            f"Matched: {matches} | Missed: {missed} | False positives: {false_positives}\n"
            f"Precision: {precision:.3f} | Recall: {recall:.3f}\n"
            "Double-click a row or use Preview Selected to inspect overlays."
        )

    def _write_summary(self, text: str):
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, text)

    def _log(self, message: str):
        self.log_queue.put(message)

    def _after(self, callback, *args):
        self.parent.after(0, lambda: callback(*args))

    def selected_result(self):
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Select a result row first.")
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.results) or self.results[index] is None:
            return None
        return self.results[index]

    def preview_selected_result(self):
        result = self.selected_result()
        if not result:
            return
        self.show_preview(result["index"])

    def show_preview(self, result_index: int):
        if result_index < 0 or result_index >= len(self.results):
            return
        self.preview_result_index = result_index
        result = self.results[result_index]
        if result is None:
            return

        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = tk.Toplevel(self.parent)
            self.preview_window.title("Inference Review Preview")
            self.preview_window.protocol("WM_DELETE_WINDOW", self.close_preview_window)
            body = ttk.Frame(self.preview_window, padding="8")
            body.pack(fill=tk.BOTH, expand=True)
            self.preview_image_label = ttk.Label(body)
            self.preview_image_label.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
            details_frame = ttk.Frame(body)
            details_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W), padx=(10, 0))
            self.preview_details_text = tk.Text(details_frame, width=48, height=32, wrap=tk.WORD)
            self.preview_details_text.pack(fill=tk.BOTH, expand=True)
            controls = ttk.Frame(self.preview_window, padding=(8, 0, 8, 8))
            controls.pack(fill=tk.X)
            ttk.Button(controls, text="Previous", command=lambda: self.move_preview(-1, failures_only=False)).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Next", command=lambda: self.move_preview(1, failures_only=False)).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Previous Failure", command=lambda: self.move_preview(-1, failures_only=True)).pack(side=tk.LEFT, padx=12)
            ttk.Button(controls, text="Next Failure", command=lambda: self.move_preview(1, failures_only=True)).pack(side=tk.LEFT, padx=3)
            ttk.Button(controls, text="Close", command=self.close_preview_window).pack(side=tk.RIGHT, padx=3)
            self.preview_window.bind("<Left>", lambda event: self.move_preview(-1, failures_only=False))
            self.preview_window.bind("<Right>", lambda event: self.move_preview(1, failures_only=False))
            self.preview_window.bind("<Escape>", lambda event: self.close_preview_window())

        self.preview_window.title(f"Inference Review Preview - {Path(result['image_path']).name}")
        self.update_preview_content(result)
        self.preview_window.focus_force()

    def close_preview_window(self):
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()
        self.preview_window = None
        self.preview_photo = None

    def move_preview(self, step: int, failures_only: bool):
        if self.preview_result_index is None:
            return
        index = self.preview_result_index + step
        while 0 <= index < len(self.results):
            result = self.results[index]
            if result is not None and (not failures_only or self.is_failure(result)):
                self.show_preview(index)
                return
            index += step

    def update_preview_content(self, result):
        image_bgr = cv2.imread(result["image_path"])
        if image_bgr is None:
            self.preview_details_text.delete(1.0, tk.END)
            self.preview_details_text.insert(tk.END, "Could not read image.")
            return

        display_bgr = self.draw_preview_overlay(image_bgr, result)
        display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_rgb)
        self.preview_photo = ImageTk.PhotoImage(pil_image)
        self.preview_image_label.configure(image=self.preview_photo)

        self.preview_details_text.delete(1.0, tk.END)
        self.preview_details_text.insert(tk.END, self.preview_details(result))
        try:
            self.results_tree.selection_set(str(result["index"]))
            self.results_tree.see(str(result["index"]))
        except tk.TclError:
            pass

    def draw_preview_overlay(self, image_bgr, result):
        max_width, max_height = 1000, 700
        h, w = image_bgr.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display = image_bgr.copy()

        thickness = max(2, int(round(min(display.shape[:2]) / 300)))

        for gt in result["gt"]:
            color = (0, 190, 0) if gt.matched else (0, 140, 255)
            label = f"GT {gt.label}"
            self.draw_labeled_box(display, gt.bbox, scale, color, label, thickness)

        for prediction in result["predictions"]:
            color = (255, 180, 0) if prediction.matched else (0, 0, 255)
            label = f"P {prediction.label} {prediction.score:.2f}"
            self.draw_labeled_box(display, prediction.bbox, scale, color, label, thickness)

        return display

    @staticmethod
    def draw_labeled_box(image, bbox, scale, color, label, thickness):
        x1, y1, x2, y2 = [int(round(value * scale)) for value in bbox]
        x1 = max(0, min(image.shape[1] - 1, x1))
        x2 = max(0, min(image.shape[1] - 1, x2))
        y1 = max(0, min(image.shape[0] - 1, y1))
        y2 = max(0, min(image.shape[0] - 1, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(image.shape[:2]) / 1100)
        font_thickness = max(1, thickness - 1)
        text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size
        text_x = x1
        text_y = y1 - 6 if y1 > text_h + baseline + 8 else y2 + text_h + 8
        text_y = max(text_h + 4, min(image.shape[0] - baseline - 2, text_y))
        bg_top = max(0, text_y - text_h - baseline - 4)
        bg_right = min(image.shape[1] - 1, text_x + text_w + 6)
        cv2.rectangle(image, (text_x, bg_top), (bg_right, text_y + baseline + 2), color, -1)
        cv2.putText(image, label, (text_x + 3, text_y), font, font_scale, (255, 255, 255), font_thickness)

    def preview_details(self, result):
        lines = [
            f"Image: {Path(result['image_path']).name}",
            f"Status: {result['status']}",
            f"XML: {Path(result['xml_path']).name if result['xml_found'] else 'missing'}",
            f"Matches: {len(result['matches'])}",
            f"Missed: {len(result['missed'])}",
            f"False positives: {len(result['false_positives'])}",
            f"Best IoU: {result['best_iou']:.3f}",
            "",
            "Ground Truth:",
        ]
        for index, gt in enumerate(result["gt"], start=1):
            match = f" matched IoU={gt.match_iou:.2f}" if gt.matched else " MISSED"
            lines.append(f"  {index}. {gt.label} {self.format_bbox(gt.bbox)}{match}")
        if not result["gt"]:
            lines.append("  none")

        lines.extend(["", "Predictions:"])
        for index, prediction in enumerate(result["predictions"], start=1):
            match = f" matched IoU={prediction.match_iou:.2f}" if prediction.matched else " FALSE POS"
            lines.append(
                f"  {index}. {prediction.label} {prediction.score:.2f} "
                f"{self.format_bbox(prediction.bbox)}{match}"
            )
        if not result["predictions"]:
            lines.append("  none")

        if result["notes"]:
            lines.extend(["", "Notes:", f"  {result['notes']}"])
        return "\n".join(lines)

    @staticmethod
    def format_bbox(bbox):
        return "[{}, {}, {}, {}]".format(*(int(round(value)) for value in bbox))

    def export_results_csv(self):
        if not self.results:
            messagebox.showinfo("Info", "Run inference before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Save inference review CSV",
            initialdir=self.dataset_dir_var.get() or self.app_root,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "image", "status", "gt_count", "prediction_count", "matches",
                        "missed", "false_positives", "best_iou", "top_prediction",
                        "notes", "image_path", "xml_path"
                    ]
                )
                writer.writeheader()
                for result in self.results:
                    if result is None:
                        continue
                    writer.writerow({
                        "image": Path(result["image_path"]).name,
                        "status": result["status"],
                        "gt_count": len(result["gt"]),
                        "prediction_count": len(result["predictions"]),
                        "matches": len(result["matches"]),
                        "missed": len(result["missed"]),
                        "false_positives": len(result["false_positives"]),
                        "best_iou": f"{result['best_iou']:.4f}",
                        "top_prediction": result["top_prediction"],
                        "notes": result["notes"],
                        "image_path": result["image_path"],
                        "xml_path": result["xml_path"],
                    })
            self._log(f"Saved inference review CSV: {path}")
            messagebox.showinfo("Success", f"Saved inference review CSV:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not export CSV: {exc}")
