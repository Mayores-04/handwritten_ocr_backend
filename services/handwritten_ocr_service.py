"""Handwritten OCR service with fallback chain: Keras CRNN -> EasyOCR."""

import logging
from typing import Any, List

import cv2
import numpy as np
import tensorflow as tf

from preprocessing.image_processors import preprocess_image
from preprocessing.image_utils import otsu_threshold
from postprocessing import post_process_handwriting


logger = logging.getLogger(__name__)


class HandwrittenOCRService:
    @staticmethod
    def _final_postprocess(result: dict) -> dict:
        """Apply postprocessing to the final output text and lines, and add debug info."""
        if not result.get("success"):
            return result
        raw_text = result.get("text", "")
        post_text = post_process_handwriting(raw_text)
        result["debug_postprocess"] = {
            "raw_text": raw_text,
            "postprocessed_text": post_text,
        }
        result["text"] = post_text
        if "lines" in result:
            result["lines"] = [post_process_handwriting(line) for line in result["lines"]]
        return result

    def __init__(self, keras_confidence_threshold: float = 0.50):
        self.keras_confidence_threshold = keras_confidence_threshold

    def _prepare_line_for_crnn(self, line_gray: np.ndarray) -> np.ndarray:
        """Resize and normalize a line image for CRNN input."""
        target_height = 32
        target_width = 256

        h, w = line_gray.shape
        if h <= 0 or w <= 0:
            padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
            norm = padded.astype("float32") / 255.0
            return norm[np.newaxis, ..., np.newaxis]

        scale = target_height / h
        new_w = max(1, int(w * scale))

        resized = cv2.resize(
            line_gray,
            (new_w, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

        if new_w < target_width:
            padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
            padded[:, :new_w] = resized
        else:
            padded = cv2.resize(
                resized,
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )

        norm = padded.astype("float32") / 255.0
        return norm[np.newaxis, ..., np.newaxis]

    def recognize(
        self,
        image: Any,
        handwriting_model: Any = None,
        char_model: Any = None,  # kept for compatibility, intentionally unused
        easyocr_reader: Any = None,
    ) -> dict[str, Any]:
        """
        Recognize handwritten text using:
        1. Keras CRNN as primary
        2. EasyOCR as the only fallback

        char_model is intentionally ignored.
        """
        try:
            if handwriting_model is None and easyocr_reader is None:
                return {
                    "success": False,
                    "error": "No OCR model loaded",
                    "text": "",
                    "engine": "none",
                }

            pil_image = preprocess_image(image)
            img_array = np.array(pil_image)

            if len(img_array.shape) == 3:
                img_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

            img_array = img_array.astype("uint8")

            thresh = otsu_threshold(img_array)
            binary = (img_array <= thresh).astype("uint8")

            # Very light morphology cleanup
            try:
                from scipy.ndimage import binary_dilation, binary_erosion

                selem = np.ones((2, 2), dtype=bool)
                binary = binary_dilation(binary, structure=selem, iterations=1).astype("uint8")
                binary = binary_erosion(binary, structure=selem, iterations=1).astype("uint8")
            except ImportError:
                pass

            lines = self._segment_lines(binary, img_array)

            crnn_result = (
                self._recognize_with_crnn(lines, handwriting_model)
                if handwriting_model is not None
                else None
            )

            easyocr_result = (
                self._recognize_with_easyocr(img_array, easyocr_reader)
                if easyocr_reader is not None
                else None
            )

            if (
                crnn_result
                and crnn_result.get("success")
                and crnn_result.get("confidence", 0.0) >= self.keras_confidence_threshold
            ):
                logger.info(
                    "[OCR] Using Keras CRNN engine (confidence: %.3f)",
                    crnn_result.get("confidence", 0.0),
                )
                best_result = crnn_result

            elif easyocr_result and easyocr_result.get("success"):
                logger.info(
                    "[OCR] Using EasyOCR fallback (confidence: %.3f)",
                    easyocr_result.get("confidence", 0.0),
                )
                best_result = easyocr_result

            else:
                logger.info("[OCR] All engines failed, returning error result.")
                best_result = {
                    "success": False,
                    "error": "All OCR engines failed",
                    "text": "",
                    "engine": "none",
                }

            def _debug_summary(result):
                if not result:
                    return None
                return {
                    "text": result.get("text"),
                    "confidence": result.get("confidence"),
                    "lines": result.get("lines"),
                    "engine": result.get("engine"),
                    "error": result.get("error") if not result.get("success", True) else None,
                }

            best_result["debug"] = {
                "crnn": _debug_summary(crnn_result),
                "easyocr": _debug_summary(easyocr_result),
            }

            return self._final_postprocess(best_result)

        except Exception as e:
            logger.error("Handwritten recognition failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": f"Recognition error: {e}",
                "text": "",
                "engine": "error",
            }

    @staticmethod
    def _correct_lines(raw_lines: List[str]) -> tuple[str, str, List[str]]:
        """Run postprocessing and return (combined_text, plain_text, corrected_lines)."""
        corrected = [post_process_handwriting(line) for line in raw_lines if line and line.strip()]
        combined = "\n".join(corrected)
        plain = " ".join(corrected)
        return combined, plain, corrected

    def _recognize_with_crnn(self, lines: List[tuple], handwriting_model: Any) -> dict[str, Any]:
        """CRNN sequence-to-sequence recognition with CTC decoding."""
        charset = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?\'\"()-"
        all_lines_text = []
        all_confs = []

        for _, line_gray in lines:
            line_input = self._prepare_line_for_crnn(line_gray)
            preds = handwriting_model.predict(line_input, verbose=0)
            pred_text, pred_conf = self._decode_crnn_prediction(preds[0], charset)

            if pred_text.strip():
                all_lines_text.append(pred_text)
                all_confs.append(pred_conf)

        if not all_lines_text:
            return {
                "success": False,
                "error": "CRNN produced no text",
                "text": "",
                "engine": "crnn",
            }

        avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
        combined_text, plain_text, corrected_lines = self._correct_lines(all_lines_text)

        return {
            "success": True,
            "text": combined_text,
            "plain_text": plain_text,
            "lines": corrected_lines,
            "confidence": avg_conf,
            "mode": "handwritten",
            "engine": "crnn",
            "line_count": len(corrected_lines),
        }

    def _decode_crnn_prediction(self, pred: np.ndarray, charset: str) -> tuple[str, float]:
        """Greedy CTC decode."""
        batch_preds = np.expand_dims(pred, axis=0)
        input_len = np.array([pred.shape[0]], dtype=np.int32)

        decoded, _ = tf.keras.backend.ctc_decode(
            batch_preds,
            input_length=input_len,
            greedy=True,
        )
        seq = decoded[0].numpy()[0]

        chars = [charset[int(idx)] for idx in seq if 0 <= idx < len(charset)]
        conf = float(np.mean(np.max(pred, axis=1))) if pred.size else 0.0

        return "".join(chars), conf

    def _recognize_with_easyocr(self, img_array: np.ndarray, easyocr_reader: Any) -> dict[str, Any]:
        """EasyOCR fallback for handwritten text with enhanced preprocessing."""
        try:
            allowlist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?\'\"()- "

            def easyocr_lines(processed_img):
                best_lines = []
                best_conf = 0.0

                for para in [True, False]:
                    results = easyocr_reader.readtext(
                        processed_img,
                        detail=1,
                        paragraph=para,
                        decoder="beamsearch",
                        min_size=2,
                        contrast_ths=0.05,
                        adjust_contrast=0.8,
                        text_threshold=0.2,
                        low_text=0.15,
                        allowlist=allowlist,
                    )

                    line_map = {}
                    for item in results:
                        if len(item) == 3:
                            bbox, text, conf = item
                        elif len(item) == 2:
                            bbox, text = item
                            conf = 1.0
                        else:
                            continue

                        text = "".join(c for c in text if c in allowlist).strip()
                        if not text or conf < 0.2:
                            continue

                        y_mid = int(np.mean([pt[1] for pt in bbox]))
                        line_key = int(round(y_mid / 15.0) * 15)
                        x_min = min(pt[0] for pt in bbox)

                        line_map.setdefault(line_key, []).append((x_min, text, conf))

                    all_lines = []
                    all_confs = []

                    for y in sorted(line_map.keys()):
                        words = sorted(line_map[y], key=lambda t: t[0])
                        line_text = " ".join(w[1] for w in words)
                        line_conf = float(np.mean([w[2] for w in words]))

                        if line_text.strip():
                            all_lines.append(line_text)
                            all_confs.append(line_conf)

                    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0

                    if (avg_conf > best_conf and all_lines) or (
                        avg_conf == best_conf and len(all_lines) > len(best_lines)
                    ):
                        best_conf = avg_conf
                        best_lines = all_lines

                return best_lines, best_conf

            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()

            processed_variants = []

            # 1. Upscaled grayscale
            upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            processed_variants.append(upscaled)

            # 2. Histogram equalization
            gentle = cv2.equalizeHist(gray)
            coords = cv2.findNonZero(gentle)
            if coords is not None:
                x, y, w_box, h_box = cv2.boundingRect(coords)
                gentle = gentle[y:y + h_box, x:x + w_box]
            processed_variants.append(gentle)

            # 3. CLAHE + adaptive threshold
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            strong = clahe.apply(gray)
            strong = cv2.adaptiveThreshold(
                strong,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                25,
                15,
            )
            coords = cv2.findNonZero(strong)
            if coords is not None:
                x, y, w_box, h_box = cv2.boundingRect(coords)
                strong = strong[y:y + h_box, x:x + w_box]
            processed_variants.append(strong)

            # 4. Otsu binarization
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(otsu)
            if coords is not None:
                x, y, w_box, h_box = cv2.boundingRect(coords)
                otsu = otsu[y:y + h_box, x:x + w_box]
            processed_variants.append(otsu)

            # 5. Contrast stretching
            p2, p98 = np.percentile(gray, (2, 98))
            contrast = np.clip((gray - p2) * 255.0 / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
            coords = cv2.findNonZero(contrast)
            if coords is not None:
                x, y, w_box, h_box = cv2.boundingRect(coords)
                contrast = contrast[y:y + h_box, x:x + w_box]
            processed_variants.append(contrast)

            best_lines = []
            best_conf = 0.0

            for variant in processed_variants:
                lines, conf = easyocr_lines(variant)
                if (conf > best_conf and lines) or (
                    conf == best_conf and len(lines) > len(best_lines)
                ):
                    best_conf = conf
                    best_lines = lines

            if not best_lines:
                return {
                    "success": False,
                    "error": "No valid lines extracted",
                    "text": "",
                    "engine": "easyocr",
                }

            combined_text, plain_text, corrected_lines = self._correct_lines(best_lines)

            return {
                "success": True,
                "text": combined_text,
                "plain_text": plain_text,
                "lines": corrected_lines,
                "confidence": best_conf,
                "mode": "handwritten",
                "engine": "easyocr",
                "line_count": len(corrected_lines),
            }

        except Exception as e:
            logger.error("EasyOCR recognition failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "engine": "easyocr",
            }

    def _segment_lines(self, binary: np.ndarray, gray: np.ndarray) -> List[tuple]:
        """Segment text lines from binary image."""
        h, _ = binary.shape
        h_proj = binary.sum(axis=1).astype(float)
        max_ink = h_proj.max()

        if max_ink == 0:
            return [(binary, gray)]

        row_threshold = max(max_ink * 0.01, 1.0)
        text_rows = h_proj > row_threshold

        diffs = np.diff(text_rows.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return [(binary, gray)]

        fragments = list(zip(starts, ends))
        heights = [e - s for s, e in fragments]
        mean_band_h = np.mean(heights) if heights else 1.0

        merged = []
        used = [False] * len(fragments)

        for i in range(len(fragments)):
            if used[i]:
                continue

            start, end = fragments[i]
            used[i] = True

            for j in range(i + 1, len(fragments)):
                if used[j]:
                    continue

                s2, e2 = fragments[j]

                if max(start, s2) < min(end, e2):
                    start, end = min(start, s2), max(end, e2)
                    used[j] = True
                elif abs(s2 - end) <= mean_band_h * 0.5 or abs(start - e2) <= mean_band_h * 0.5:
                    start, end = min(start, s2), max(end, e2)
                    used[j] = True

            merged.append((start, end))

        pad = max(int(mean_band_h * 0.10), 2)
        lines = []

        for r0, r1 in merged:
            r0p = max(0, r0 - pad)
            r1p = min(h, r1 + pad)

            line_bin = binary[r0p:r1p, :]
            line_gray = gray[r0p:r1p, :]

            if line_bin.shape[0] < 3 or line_bin.sum() == 0:
                continue

            lines.append((line_bin, line_gray))

        return lines if lines else [(binary, gray)]
