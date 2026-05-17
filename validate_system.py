"""Local validation helper for the OCR system.

Checks:
- real dataset discovery
- Keras/TensorFlow model loading status
- backend /api/health
- optional OCR request against /api/ocr/handwritten
- optional frontend availability
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

from utils.env_loader import load_env_local

load_env_local()

from dataset_loader import discover_labeled_image_samples
from gpu_utils import get_tensorflow_runtime_status
from services.model_service import model_service


def print_check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def get_json(url: str, timeout: int = 20) -> tuple[bool, dict | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return True, json.loads(body), ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, None, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, None, str(exc)


def post_multipart(url: str, image_path: Path, mode: str, timeout: int = 60) -> tuple[bool, dict | None, str]:
    boundary = "----ocr-validation-boundary"
    mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    image_bytes = image_path.read_bytes()

    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="mode"\r\n\r\n'
            f"{mode}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8"),
        image_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, payload, ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return False, json.loads(body), f"HTTP {exc.code}"
        except Exception:
            return False, None, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local OCR system.")
    parser.add_argument("--dataset-path", default="data", help="Backend dataset path")
    parser.add_argument("--backend-url", default="http://localhost:5000", help="Backend base URL")
    parser.add_argument("--frontend-url", default="http://localhost:3000", help="Frontend base URL")
    parser.add_argument("--image", default=None, help="Optional image path for an OCR request")
    parser.add_argument("--skip-http", action="store_true", help="Skip backend/frontend HTTP checks")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if TensorFlow cannot see a GPU")
    args = parser.parse_args()

    failures = 0

    dataset = discover_labeled_image_samples(args.dataset_path, max_samples=20)
    dataset_ok = dataset.total_samples > 0
    print_check(
        "Dataset detected",
        dataset_ok,
        f"{dataset.total_samples} sample(s) via {dataset.source}",
    )
    if dataset.warnings:
        for warning in dataset.warnings:
            print(f"  warning: {warning}")
    if not dataset_ok:
        failures += 1

    runtime_status = get_tensorflow_runtime_status()
    gpu_ok = bool(runtime_status.get("gpu_available"))
    detail = runtime_status.get("error") or ", ".join(runtime_status.get("gpu_names", [])) or "CPU only"
    print_check("TensorFlow GPU visible", gpu_ok, detail)
    if args.require_gpu and not gpu_ok:
        failures += 1

    model_service.warmup_models()
    model_status = model_service.get_status()
    model_ok = bool(model_status.get("handwriting_ready"))
    print_check(
        "Keras handwriting model loaded",
        model_ok,
        model_status.get("models", {}).get("handwriting_model", {}).get("path") or model_status.get("load_error") or "",
    )
    if not model_ok:
        failures += 1

    if args.skip_http:
        return failures

    backend_ok, health, backend_error = get_json(f"{args.backend_url.rstrip('/')}/api/health")
    print_check("Backend /api/health", backend_ok, backend_error if not backend_ok else health.get("status", ""))
    if not backend_ok:
        failures += 1

    frontend_ok = True
    try:
        with urllib.request.urlopen(args.frontend_url, timeout=15) as response:
            frontend_ok = 200 <= response.status < 500
        print_check("Frontend reachable", frontend_ok, args.frontend_url)
    except Exception as exc:
        print_check("Frontend reachable", False, str(exc))
        failures += 1

    image_path = Path(args.image).resolve() if args.image else None
    if image_path is None and dataset.train_samples:
        image_path = dataset.train_samples[0].image_path

    if image_path and image_path.exists() and backend_ok:
        ocr_ok, payload, ocr_error = post_multipart(
            f"{args.backend_url.rstrip('/')}/api/ocr/handwritten",
            image_path,
            "handwritten",
        )
        success = bool(ocr_ok and payload and payload.get("success"))
        detail = ""
        if payload:
            detail = (
                f"engine={payload.get('engine')} mode_used={payload.get('mode_used')} "
                f"confidence={payload.get('confidence')}"
            )
        else:
            detail = ocr_error
        print_check("OCR extraction", success, detail)
        if not success:
            failures += 1

    return failures


if __name__ == "__main__":
    sys.exit(main())
