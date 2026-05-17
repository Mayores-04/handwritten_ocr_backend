"""Keras/TensorFlow model loading and status reporting."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from gpu_utils import get_tensorflow_runtime_status

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BACKEND_ROOT.parent
MODEL_DIR = BACKEND_ROOT / "models"
BACKUP_MODEL_DIR = WORKSPACE_ROOT / "large_artifacts_backup" / "models"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _candidate_paths(env_var: str, *paths: Path) -> list[Path]:
    candidates: list[Path] = []
    env_value = _env_path(env_var)
    if env_value:
        candidates.append(env_value)
    candidates.extend(paths)
    return candidates


HANDWRITING_MODEL_CANDIDATES = _candidate_paths(
    "OCR_HANDWRITING_MODEL_PATH",
    MODEL_DIR / "handwriting_model.keras",
    BACKUP_MODEL_DIR / "handwriting_model.keras",
)

CHAR_MODEL_CANDIDATES = _candidate_paths(
    "OCR_CHAR_MODEL_PATH",
    MODEL_DIR / "char_model.keras",
    BACKUP_MODEL_DIR / "char_model.keras",
)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _shape_to_json(shape: Any) -> Any:
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        return [_shape_to_json(item) for item in shape]
    try:
        return int(shape)
    except Exception:
        return str(shape)


def _make_ctc_loss(tf_module: Any):
    def ctc_loss_fn(y_true, y_pred):
        labels, label_lengths, input_lengths = y_true
        y_pred = tf_module.cast(y_pred, tf_module.float32)
        labels = tf_module.cast(labels, tf_module.int32)
        label_lengths = tf_module.cast(tf_module.reshape(label_lengths, [-1]), tf_module.int32)
        input_lengths = tf_module.cast(tf_module.reshape(input_lengths, [-1]), tf_module.int32)

        batch_size = tf_module.shape(labels)[0]
        max_label_length = tf_module.shape(labels)[1]
        mask = tf_module.sequence_mask(label_lengths, maxlen=max_label_length)
        sparse_indices = tf_module.where(mask)
        sparse_values = tf_module.gather_nd(labels, sparse_indices)
        sparse_labels = tf_module.SparseTensor(
            indices=tf_module.cast(sparse_indices, tf_module.int64),
            values=sparse_values,
            dense_shape=tf_module.cast([batch_size, max_label_length], tf_module.int64),
        )
        logits = tf_module.math.log(tf_module.clip_by_value(y_pred, 1e-7, 1.0))
        num_classes = y_pred.shape[-1]
        if num_classes is None:
            raise ValueError("CTC loss requires a statically known number of output classes.")
        return tf_module.nn.ctc_loss(
            labels=sparse_labels,
            logits=logits,
            label_length=label_lengths,
            logit_length=input_lengths,
            logits_time_major=False,
            blank_index=int(num_classes) - 1,
        )

    ctc_loss_fn.__name__ = "ctc_loss_fn"
    return ctc_loss_fn


class ModelService:
    """Lazy loader for local Keras OCR models."""

    def __init__(self):
        self._keras_char_model: Any | None = None
        self._keras_handwriting_model: Any | None = None
        self._char_model_path: Path | None = None
        self._handwriting_model_path: Path | None = None
        self._models_warmed = False
        self._load_attempted = False
        self._use_keras = False
        self._keras_version: str | None = None
        self._tensorflow_version: str | None = None
        self._load_error: str | None = None

    def _import_keras(self) -> tuple[Any | None, Any | None]:
        try:
            import tensorflow as tf
            import keras

            self._tensorflow_version = getattr(tf, "__version__", "unknown")
            self._keras_version = getattr(keras, "__version__", "unknown")
            return keras, tf
        except Exception as exc:
            self._load_error = (
                "TensorFlow/Keras could not be imported. Recreate the backend "
                "virtual environment and install requirements.txt. "
                f"Original error: {exc}"
            )
            logger.error(self._load_error)
            return None, None

    def _load_single_model(
        self,
        keras_module: Any,
        tf_module: Any,
        model_name: str,
        model_path: Path | None,
    ) -> Any | None:
        if model_path is None:
            logger.warning("%s model file was not found.", model_name)
            return None

        try:
            custom_objects = {
                "ctc_loss_fn": _make_ctc_loss(tf_module),
                "Custom>ctc_loss_fn": _make_ctc_loss(tf_module),
            }

            try:
                model = keras_module.models.load_model(
                    str(model_path),
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False,
                )
            except TypeError:
                model = keras_module.models.load_model(
                    str(model_path),
                    custom_objects=custom_objects,
                    compile=False,
                )

            logger.info(
                "Loaded %s model from %s | input=%s | output=%s",
                model_name,
                model_path,
                getattr(model, "input_shape", None),
                getattr(model, "output_shape", None),
            )
            return model
        except Exception as exc:
            logger.exception("Failed to load %s model from %s", model_name, model_path)
            self._load_error = f"Failed to load {model_name} model from {model_path}: {exc}"
            return None

    def load_keras_models(self, force_reload: bool = False) -> bool:
        """Load available Keras models. Handwriting CRNN is the primary model."""

        if self._load_attempted and not force_reload:
            return self._use_keras

        if force_reload:
            self._keras_char_model = None
            self._keras_handwriting_model = None
            self._char_model_path = None
            self._handwriting_model_path = None

        self._load_attempted = True
        self._load_error = None

        keras_module, tf_module = self._import_keras()
        if keras_module is None or tf_module is None:
            self._use_keras = False
            return False

        self._handwriting_model_path = _first_existing(HANDWRITING_MODEL_CANDIDATES)
        self._char_model_path = _first_existing(CHAR_MODEL_CANDIDATES)

        if self._handwriting_model_path and self._handwriting_model_path.parent == BACKUP_MODEL_DIR:
            logger.warning(
                "Using backup handwriting model outside backend/models: %s",
                self._handwriting_model_path,
            )

        self._keras_handwriting_model = self._load_single_model(
            keras_module,
            tf_module,
            "handwriting CRNN",
            self._handwriting_model_path,
        )
        self._keras_char_model = self._load_single_model(
            keras_module,
            tf_module,
            "character CNN",
            self._char_model_path,
        )

        self._use_keras = self._keras_handwriting_model is not None or self._keras_char_model is not None
        if self._keras_handwriting_model is not None:
            logger.info("Keras handwriting CRNN is ready and will be tried before EasyOCR.")
        elif self._keras_char_model is not None:
            logger.warning(
                "Only char_model.keras is loaded. It is a character classifier and is not enough "
                "for full handwritten word/line OCR."
            )
        else:
            logger.error("No usable Keras models were loaded.")

        return self._use_keras

    def warmup_models(self) -> bool:
        if self._models_warmed:
            return self._use_keras

        logger.info("Warming Keras/TensorFlow OCR models...")
        loaded = self.load_keras_models()
        self._models_warmed = True
        return loaded

    def get_keras_char_model(self) -> Any | None:
        if self._keras_char_model is None and not self._load_attempted:
            self.load_keras_models()
        return self._keras_char_model

    def get_keras_handwriting_model(self) -> Any | None:
        if self._keras_handwriting_model is None and not self._load_attempted:
            self.load_keras_models()
        return self._keras_handwriting_model

    def get_status(self) -> dict[str, Any]:
        handwriting_loaded = self._keras_handwriting_model is not None
        char_loaded = self._keras_char_model is not None

        return {
            "keras_ready": handwriting_loaded or char_loaded,
            "handwriting_ready": handwriting_loaded,
            "char_model_ready": char_loaded,
            "using_keras": self._use_keras,
            "warmed": self._models_warmed,
            "primary_engine": "Keras/TensorFlow CRNN" if handwriting_loaded else "EasyOCR fallback",
            "tensorflow_version": self._tensorflow_version,
            "keras_version": self._keras_version,
            "tensorflow_runtime": get_tensorflow_runtime_status(),
            "load_error": self._load_error,
            "models": {
                "handwriting_model": {
                    "loaded": handwriting_loaded,
                    "path": str(self._handwriting_model_path) if self._handwriting_model_path else None,
                    "input_shape": _shape_to_json(
                        getattr(self._keras_handwriting_model, "input_shape", None)
                    ),
                    "output_shape": _shape_to_json(
                        getattr(self._keras_handwriting_model, "output_shape", None)
                    ),
                    "candidates": [str(path) for path in HANDWRITING_MODEL_CANDIDATES],
                },
                "char_model": {
                    "loaded": char_loaded,
                    "path": str(self._char_model_path) if self._char_model_path else None,
                    "input_shape": _shape_to_json(getattr(self._keras_char_model, "input_shape", None)),
                    "output_shape": _shape_to_json(getattr(self._keras_char_model, "output_shape", None)),
                    "candidates": [str(path) for path in CHAR_MODEL_CANDIDATES],
                },
            },
        }


model_service = ModelService()
