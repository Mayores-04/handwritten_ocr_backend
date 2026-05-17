"""Handwritten OCR pipeline.

The Keras CRNN model is tried first. EasyOCR is loaded only when the Keras
model is missing, fails, or returns a low-confidence result.
"""

from __future__ import annotations

import logging
from typing import Any, List

import cv2
import numpy as np

from preprocessing.image_processors import preprocess_image
from preprocessing.image_utils import otsu_threshold
from postprocessing import post_process_handwriting, process_lines
from .printed_ocr_service import PrintedOCRService

logger = logging.getLogger(__name__)

CRNN_CHARSET = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?\'\"()-"


class HandwrittenOCRService:
    def __init__(self, keras_confidence_threshold: float = 0.50):
        self.keras_confidence_threshold = keras_confidence_threshold

    @staticmethod
    def _final_postprocess(result: dict) -> dict:
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
            result["lines"] = process_lines(result["lines"], mode="handwriting")

        return result

    def recognize(
        self,
        image: Any,
        handwriting_model: Any = None,
        char_model: Any = None,
        easyocr_reader: Any = None,
        easyocr_loader: Any = None,
    ) -> dict[str, Any]:
        """Recognize handwriting using Keras first, then EasyOCR fallback."""

        _ = char_model  # char_model is intentionally not used for word/line OCR.

        try:
            if handwriting_model is None and easyocr_reader is None and easyocr_loader is None:
                return {
                    "success": False,
                    "error": "No OCR model loaded",
                    "text": "",
                    "confidence": 0.0,
                    "engine": "none",
                }

            pil_image = preprocess_image(image)
            img_array = np.array(pil_image)
            gray = self._to_grayscale(img_array)
            binary = self._binarize(gray)
            lines = self._segment_lines(binary, gray)

            crnn_result = (
                self._recognize_with_crnn(lines, handwriting_model)
                if handwriting_model is not None
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
                crnn_result["fallback_used"] = False
                return self._final_postprocess(crnn_result)

            if crnn_result and crnn_result.get("success"):
                logger.warning(
                    "[OCR] Keras CRNN confidence %.3f is below threshold %.3f; trying EasyOCR fallback",
                    crnn_result.get("confidence", 0.0),
                    self.keras_confidence_threshold,
                )

            if easyocr_reader is None and easyocr_loader is not None:
                easyocr_reader = easyocr_loader()

            easyocr_result = (
                self._recognize_with_easyocr(gray, easyocr_reader)
                if easyocr_reader is not None
                else None
            )

            printed_result = None
            if easyocr_reader is not None:
                try:
                    printed_result = PrintedOCRService().recognize(pil_image, easyocr_reader)
                except Exception:
                    printed_result = None

            best_result = self._choose_fallback_result(easyocr_result, printed_result)
            if best_result:
                logger.info(
                    "[OCR] Using EasyOCR fallback (confidence: %.3f)",
                    best_result.get("confidence", 0.0),
                )
                best_result["fallback_used"] = True
            else:
                best_result = {
                    "success": False,
                    "error": "All OCR engines failed",
                    "text": "",
                    "confidence": 0.0,
                    "engine": "none",
                }

            best_result["debug"] = {
                "crnn": self._debug_summary(crnn_result),
                "easyocr": self._debug_summary(easyocr_result),
                "printed_grouping": self._debug_summary(printed_result),
            }

            return self._final_postprocess(best_result)

        except Exception as exc:
            logger.error("Handwritten recognition failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": f"Recognition error: {exc}",
                "text": "",
                "confidence": 0.0,
                "engine": "error",
            }

    @staticmethod
    def _to_grayscale(img_array: np.ndarray) -> np.ndarray:
        if len(img_array.shape) == 3:
            return np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype("uint8")
        return img_array.astype("uint8")

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        thresh = otsu_threshold(gray)
        binary = (gray <= thresh).astype("uint8")

        try:
            from scipy.ndimage import binary_dilation, binary_erosion

            selem = np.ones((2, 2), dtype=bool)
            binary = binary_dilation(binary, structure=selem, iterations=1).astype("uint8")
            binary = binary_erosion(binary, structure=selem, iterations=1).astype("uint8")
        except ImportError:
            pass

        return binary

    @staticmethod
    def _debug_summary(result: dict | None) -> dict | None:
        if not result:
            return None
        return {
            "text": result.get("text"),
            "confidence": result.get("confidence"),
            "lines": result.get("lines"),
            "engine": result.get("engine"),
            "error": result.get("error") if not result.get("success", True) else None,
        }

    @staticmethod
    def _choose_fallback_result(easyocr_result: dict | None, printed_result: dict | None) -> dict | None:
        if not easyocr_result or not easyocr_result.get("success"):
            return printed_result if printed_result and printed_result.get("success") else None

        chosen = easyocr_result
        if printed_result and printed_result.get("success"):
            p_lines = printed_result.get("lines", [])
            e_lines = easyocr_result.get("lines", [])
            p_conf = printed_result.get("confidence", 0.0)
            e_conf = easyocr_result.get("confidence", 0.0)

            if p_lines and (len(p_lines) > len(e_lines) or p_conf - e_conf > 0.03):
                chosen = printed_result

        return chosen

    @staticmethod
    def _correct_lines(raw_lines: List[str]) -> tuple[str, str, List[str]]:
        corrected = process_lines(raw_lines, mode="handwriting")
        combined = "\n".join(corrected)
        plain = " ".join(corrected)
        return combined, plain, corrected

    def _prepare_crop_for_crnn(self, gray_crop: np.ndarray) -> np.ndarray:
        target_height = 32
        target_width = 512

        h, w = gray_crop.shape
        if h <= 0 or w <= 0:
            padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
            return (padded.astype("float32") / 255.0)[np.newaxis, ..., np.newaxis]

        scale = target_height / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(gray_crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        if new_w < target_width:
            padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
            padded[:, :new_w] = resized
        else:
            padded = cv2.resize(resized, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        return (padded.astype("float32") / 255.0)[np.newaxis, ..., np.newaxis]

    def _recognize_with_crnn(self, lines: List[tuple], handwriting_model: Any) -> dict[str, Any]:
        all_lines_text: list[str] = []
        all_confs: list[float] = []

        for line_binary, line_gray in lines:
            word_crops = self._segment_words(line_binary, line_gray)
            line_words = []
            line_confs = []

            for word_gray in word_crops:
                model_input = self._prepare_crop_for_crnn(word_gray)
                preds = handwriting_model.predict(model_input, verbose=0)
                pred_text, pred_conf = self._decode_crnn_prediction(preds[0], CRNN_CHARSET)

                pred_text = pred_text.strip()
                if pred_text:
                    line_words.append(pred_text)
                    line_confs.append(pred_conf)

            if line_words:
                all_lines_text.append(" ".join(line_words))
                all_confs.extend(line_confs)

        if not all_lines_text:
            return {
                "success": False,
                "error": "CRNN produced no text",
                "text": "",
                "confidence": 0.0,
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
            "mode_used": "keras_handwriting",
            "line_count": len(corrected_lines),
        }

    @staticmethod
    def _decode_crnn_prediction(pred: np.ndarray, charset: str) -> tuple[str, float]:
        if pred.size == 0:
            return "", 0.0

        blank_index = len(charset)
        indices = np.argmax(pred, axis=1)
        max_probs = np.max(pred, axis=1)

        chars: list[str] = []
        char_confs: list[float] = []
        prev_idx: int | None = None

        for idx, prob in zip(indices, max_probs):
            idx_int = int(idx)
            if idx_int == prev_idx:
                continue
            prev_idx = idx_int

            if idx_int == blank_index:
                continue
            if 0 <= idx_int < len(charset):
                chars.append(charset[idx_int])
                char_confs.append(float(prob))

        conf = float(np.mean(char_confs)) if char_confs else 0.0
        return "".join(chars), conf

    def _recognize_with_easyocr(self, gray: np.ndarray, easyocr_reader: Any) -> dict[str, Any]:
        try:
            allowlist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?\'\"()- "

            processed_variants = self._easyocr_variants(gray)
            best_lines: list[str] = []
            best_conf = 0.0

            for variant in processed_variants:
                lines, conf = self._read_easyocr_lines(variant, easyocr_reader, allowlist)
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
                    "confidence": 0.0,
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
                "mode_used": "handwriting_easyocr",
                "line_count": len(corrected_lines),
            }

        except Exception as exc:
            logger.error("EasyOCR recognition failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": str(exc),
                "text": "",
                "confidence": 0.0,
                "engine": "easyocr",
            }

    @staticmethod
    def _easyocr_variants(gray: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = []

        variants.append(cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.equalizeHist(gray))

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
        variants.append(strong)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        p2, p98 = np.percentile(gray, (2, 98))
        contrast = np.clip((gray - p2) * 255.0 / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
        variants.append(contrast)

        return [HandwrittenOCRService._crop_nonwhite(variant) for variant in variants]

    @staticmethod
    def _crop_nonwhite(image: np.ndarray) -> np.ndarray:
        coords = cv2.findNonZero(255 - image if np.mean(image) > 127 else image)
        if coords is None:
            return image
        x, y, w_box, h_box = cv2.boundingRect(coords)
        return image[y:y + h_box, x:x + w_box]

    @staticmethod
    def _read_easyocr_lines(
        processed_img: np.ndarray,
        easyocr_reader: Any,
        allowlist: str,
    ) -> tuple[list[str], float]:
        best_lines: list[str] = []
        best_conf = 0.0

        for para in (True, False):
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

            line_map: dict[int, list[tuple[float, str, float]]] = {}
            for item in results:
                if len(item) == 3:
                    bbox, text, conf = item
                elif len(item) == 2:
                    bbox, text = item
                    conf = 1.0
                else:
                    continue

                text = "".join(c for c in str(text) if c in allowlist).strip()
                if not text or conf < 0.2:
                    continue

                y_mid = int(np.mean([pt[1] for pt in bbox]))
                line_key = int(round(y_mid / 15.0) * 15)
                x_min = min(pt[0] for pt in bbox)
                line_map.setdefault(line_key, []).append((x_min, text, float(conf)))

            lines = []
            confs = []
            for y in sorted(line_map.keys()):
                words = sorted(line_map[y], key=lambda item: item[0])
                line_text = " ".join(word[1] for word in words)
                line_conf = float(np.mean([word[2] for word in words]))
                if line_text.strip():
                    lines.append(line_text)
                    confs.append(line_conf)

            avg_conf = float(np.mean(confs)) if confs else 0.0
            if (avg_conf > best_conf and lines) or (
                avg_conf == best_conf and len(lines) > len(best_lines)
            ):
                best_conf = avg_conf
                best_lines = lines

        return best_lines, best_conf

    def _segment_lines(self, binary: np.ndarray, gray: np.ndarray) -> List[tuple[np.ndarray, np.ndarray]]:
        h, _ = binary.shape
        horizontal_projection = binary.sum(axis=1).astype(float)
        max_ink = horizontal_projection.max()

        if max_ink == 0:
            return [(binary, gray)]

        row_threshold = max(max_ink * 0.01, 1.0)
        text_rows = horizontal_projection > row_threshold

        starts, ends = self._runs_from_mask(text_rows)
        if not starts:
            return [(binary, gray)]

        fragments = list(zip(starts, ends))
        heights = [end - start for start, end in fragments]
        mean_band_h = float(np.mean(heights)) if heights else 1.0
        merged = self._merge_runs(fragments, max_gap=max(int(mean_band_h * 0.5), 2))

        pad = max(int(mean_band_h * 0.10), 2)
        lines = []

        for row_start, row_end in merged:
            r0 = max(0, row_start - pad)
            r1 = min(h, row_end + pad)
            line_binary = binary[r0:r1, :]
            line_gray = gray[r0:r1, :]

            if line_binary.shape[0] >= 3 and line_binary.sum() > 0:
                lines.append((line_binary, line_gray))

        return lines if lines else [(binary, gray)]

    def _segment_words(self, line_binary: np.ndarray, line_gray: np.ndarray) -> list[np.ndarray]:
        h, w = line_binary.shape
        vertical_projection = line_binary.sum(axis=0).astype(float)
        max_ink = vertical_projection.max()

        if max_ink == 0:
            return [line_gray]

        col_threshold = max(max_ink * 0.03, 1.0)
        text_cols = vertical_projection > col_threshold
        starts, ends = self._runs_from_mask(text_cols)

        if not starts:
            return [line_gray]

        fragments = list(zip(starts, ends))
        merge_gap = max(3, int(h * 0.22))
        merged = self._merge_runs(fragments, max_gap=merge_gap)

        crops: list[np.ndarray] = []
        pad_x = max(2, int(h * 0.08))
        pad_y = max(1, int(h * 0.05))

        for col_start, col_end in merged:
            if col_end - col_start < 2:
                continue
            c0 = max(0, col_start - pad_x)
            c1 = min(w, col_end + pad_x)

            crop_binary = line_binary[:, c0:c1]
            row_projection = crop_binary.sum(axis=1)
            text_rows = np.where(row_projection > 0)[0]
            if text_rows.size == 0:
                continue

            r0 = max(0, int(text_rows[0]) - pad_y)
            r1 = min(h, int(text_rows[-1]) + pad_y + 1)
            crop = line_gray[r0:r1, c0:c1]

            if crop.shape[0] >= 3 and crop.shape[1] >= 2:
                crops.append(crop)

        return crops if crops else [line_gray]

    @staticmethod
    def _runs_from_mask(mask: np.ndarray) -> tuple[list[int], list[int]]:
        diffs = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0].tolist()
        ends = np.where(diffs == -1)[0].tolist()
        return starts, ends

    @staticmethod
    def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
        if not runs:
            return []

        merged = [runs[0]]
        for start, end in runs[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return merged
