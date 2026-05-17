"""TensorFlow runtime helpers for GPU-aware training and validation."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TensorFlowRuntimeInfo:
    tensorflow_version: str | None
    keras_version: str | None
    python_version: str
    system: str
    gpu_names: list[str]
    mixed_precision: bool
    xla: bool
    error: str | None = None

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_names)

    @property
    def gpu_available(self) -> bool:
        return self.gpu_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensorflow_version": self.tensorflow_version,
            "keras_version": self.keras_version,
            "python_version": self.python_version,
            "system": self.system,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "mixed_precision": self.mixed_precision,
            "xla": self.xla,
            "error": self.error,
        }


def _windows_gpu_guidance() -> str:
    return (
        "TensorFlow 2.11+ does not provide official native-Windows CUDA GPU "
        "support. For fast NVIDIA GPU training, run this backend inside WSL2 "
        "and install tensorflow[and-cuda]. Native Windows can still run CPU "
        "TensorFlow, or older TensorFlow 2.10 GPU with an older Python stack."
    )


def configure_tensorflow_runtime(
    tf_module,
    *,
    require_gpu: bool = False,
    enable_mixed_precision: bool = False,
    enable_xla: bool = False,
    memory_growth: bool = True,
    log_devices: bool = True,
) -> TensorFlowRuntimeInfo:
    """Configure TensorFlow before model creation/training."""

    try:
        keras_version = getattr(tf_module.keras, "__version__", None)
        gpus = tf_module.config.list_physical_devices("GPU")

        if memory_growth:
            for gpu in gpus:
                try:
                    tf_module.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as exc:
                    logger.warning("Could not set GPU memory growth for %s: %s", gpu.name, exc)

        mixed_precision_enabled = False
        if enable_mixed_precision and gpus:
            tf_module.keras.mixed_precision.set_global_policy("mixed_float16")
            mixed_precision_enabled = True
            logger.info("Mixed precision enabled: mixed_float16")
        elif enable_mixed_precision:
            logger.warning("Mixed precision requested, but no TensorFlow GPU is visible.")

        xla_enabled = False
        if enable_xla:
            tf_module.config.optimizer.set_jit(True)
            xla_enabled = True
            logger.info("XLA JIT enabled")

        gpu_names = [gpu.name for gpu in gpus]
        info = TensorFlowRuntimeInfo(
            tensorflow_version=getattr(tf_module, "__version__", None),
            keras_version=keras_version,
            python_version=platform.python_version(),
            system=f"{platform.system()} {platform.release()}",
            gpu_names=gpu_names,
            mixed_precision=mixed_precision_enabled,
            xla=xla_enabled,
        )

        if log_devices:
            logger.info("TensorFlow version: %s", info.tensorflow_version)
            logger.info("Keras version: %s", info.keras_version)
            logger.info("Python version: %s", info.python_version)
            if gpu_names:
                logger.info("TensorFlow GPU(s): %s", ", ".join(gpu_names))
            else:
                logger.warning("TensorFlow sees no GPU devices.")
                if platform.system().lower() == "windows":
                    logger.warning(_windows_gpu_guidance())

        if require_gpu and not gpu_names:
            raise RuntimeError(
                "GPU training was required, but TensorFlow sees no GPU devices. "
                + (_windows_gpu_guidance() if platform.system().lower() == "windows" else "")
            )

        return info
    except Exception as exc:
        if require_gpu:
            raise
        logger.warning("TensorFlow runtime configuration failed: %s", exc)
        return TensorFlowRuntimeInfo(
            tensorflow_version=getattr(tf_module, "__version__", None),
            keras_version=getattr(getattr(tf_module, "keras", None), "__version__", None),
            python_version=platform.python_version(),
            system=f"{platform.system()} {platform.release()}",
            gpu_names=[],
            mixed_precision=False,
            xla=False,
            error=str(exc),
        )


def get_tensorflow_runtime_status() -> dict[str, Any]:
    """Return TensorFlow/GPU status without forcing the app to crash."""

    try:
        import tensorflow as tf
    except Exception as exc:
        return {
            "tensorflow_imported": False,
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "error": str(exc),
        }

    info = configure_tensorflow_runtime(tf, log_devices=False)
    payload = info.to_dict()
    payload["tensorflow_imported"] = True
    return payload
