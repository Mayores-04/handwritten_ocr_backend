"""Printed OCR service using EasyOCR with preprocessing and spatial sorting."""

import logging
from typing import Any, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from preprocessing.image_processors import preprocess_image

logger = logging.getLogger(__name__)


class PrintedOCRService:
    """Printed text OCR pipeline (EasyOCR)."""

    @staticmethod
    def _bbox_top(det: tuple) -> float:
        return float(min(pt[1] for pt in det[0]))

    @staticmethod
    def _bbox_bottom(det: tuple) -> float:
        return float(max(pt[1] for pt in det[0]))

    @staticmethod
    def _bbox_left(det: tuple) -> float:
        return float(min(pt[0] for pt in det[0]))

    @staticmethod
    def _bbox_right(det: tuple) -> float:
        return float(max(pt[0] for pt in det[0]))

    def _normalize_detection(self, detection: tuple) -> dict[str, Any]:
        text = str(detection[1]).replace('\n', ' ').strip()
        left = self._bbox_left(detection)
        right = self._bbox_right(detection)
        top = self._bbox_top(detection)
        bottom = self._bbox_bottom(detection)
        return {
            'text': text,
            'conf': float(detection[2]),
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'center_y': (top + bottom) / 2.0,
            'height': max(bottom - top, 1.0),
            'char_width': max((right - left) / max(len(text), 1), 1.0),
        }

    def _preprocess_for_easyocr(self, pil_image: Image.Image) -> np.ndarray:
        """
        Prepare a PIL image for EasyOCR:
        1. Convert to RGB numpy
        2. Upscale short images so EasyOCR has enough resolution
        3. Mild sharpening + contrast boost
        """
        img = pil_image.convert('RGB')
        w, h = img.size

        min_dim = min(w, h)
        if min_dim < 800:
            scale = 800 / min_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "_preprocess_for_easyocr: upscaled %dx%d -> %dx%d (x%.2f)",
                w,
                h,
                new_w,
                new_h,
                scale,
            )

        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(1.4)
        return np.array(img)

    def _sort_detections_spatially(self, results: list, line_height_tolerance: float = 0.6) -> list:
        """Sort detections in reading order (top-to-bottom, left-to-right)."""
        if not results:
            return results

        sorted_dets = sorted(results, key=self._bbox_top)

        rows = []
        used = [False] * len(sorted_dets)

        for i, det in enumerate(sorted_dets):
            if used[i]:
                continue

            row = [det]
            used[i] = True
            cy_i = (self._bbox_top(det) + self._bbox_bottom(det)) / 2.0
            h_i = max(self._bbox_bottom(det) - self._bbox_top(det), 1)

            for j in range(i + 1, len(sorted_dets)):
                if used[j]:
                    continue
                cy_j = (self._bbox_top(sorted_dets[j]) + self._bbox_bottom(sorted_dets[j])) / 2.0
                h_j = max(self._bbox_bottom(sorted_dets[j]) - self._bbox_top(sorted_dets[j]), 1)
                tol = max(h_i, h_j) * line_height_tolerance
                if abs(cy_i - cy_j) <= tol:
                    row.append(sorted_dets[j])
                    used[j] = True

            row.sort(key=self._bbox_left)
            rows.append(row)

        return [det for row in rows for det in row]

    def _group_detections_into_lines(
        self,
        detections: List[dict[str, Any]],
        line_height_tolerance: float = 0.7,
    ) -> List[List[dict[str, Any]]]:
        """Cluster token detections into visual text lines."""
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: (d['center_y'], d['left']))
        lines: List[List[dict[str, Any]]] = []

        for det in sorted_dets:
            assigned = False
            for line in lines:
                line_center = float(np.mean([d['center_y'] for d in line]))
                line_height = float(np.median([d['height'] for d in line]))
                tol = max(line_height, det['height']) * line_height_tolerance
                if abs(det['center_y'] - line_center) <= tol:
                    line.append(det)
                    assigned = True
                    break

            if not assigned:
                lines.append([det])

        for line in lines:
            line.sort(key=lambda d: d['left'])

        lines.sort(key=lambda line: min(d['top'] for d in line))
        return lines

    def _compose_line_text(self, line_detections: List[dict[str, Any]]) -> str:
        """Join tokens from one line while preserving visible horizontal gaps."""
        if not line_detections:
            return ''

        char_width = float(np.median([d['char_width'] for d in line_detections]))
        char_width = max(char_width, 1.0)

        parts: List[str] = []
        prev_right: float | None = None

        for det in line_detections:
            token = det['text']
            if not token:
                continue

            if prev_right is None:
                parts.append(token)
                prev_right = det['right']
                continue

            gap = det['left'] - prev_right
            if gap <= char_width * 1.2:
                spacer = ''
            elif gap <= char_width * 4.0:
                spacer = ' '
            elif gap <= char_width * 8.0:
                spacer = '  '
            else:
                spacer = '    '

            parts.append(spacer + token)
            prev_right = det['right']

        return ''.join(parts).strip()

    def recognize(self, image: Any, easyocr_reader: Any) -> dict[str, Any]:
        """Printed OCR using EasyOCR with preprocessing + spatial ordering."""
        try:
            pil_image = preprocess_image(image)
            img_array = self._preprocess_for_easyocr(pil_image)

            results = easyocr_reader.readtext(img_array, paragraph=False)
            if not results:
                results = easyocr_reader.readtext(np.array(pil_image), paragraph=False)

            if not results:
                return {'success': False, 'error': 'No text detected', 'text': ''}

            results = self._sort_detections_spatially(results)

            min_detection_conf = 0.30
            results = [r for r in results if float(r[2]) >= min_detection_conf]

            if not results:
                return {'success': False, 'error': 'No confident text detected', 'text': ''}

            normalized_detections = [self._normalize_detection(d) for d in results]
            normalized_detections = [d for d in normalized_detections if d['text']]

            line_groups = self._group_detections_into_lines(normalized_detections)

            text_lines = [self._compose_line_text(group) for group in line_groups]
            text_lines = [line for line in text_lines if line]
            all_confs = [d['conf'] for d in normalized_detections]

            if not text_lines:
                return {'success': False, 'error': 'No text detected after filtering', 'text': ''}

            final_text = '\n'.join(text_lines)
            plain_text = ' '.join(text_lines)
            avg_confidence = float(np.mean(all_confs)) if all_confs else 0.0
            visual_line_count = len(text_lines)

            if avg_confidence < 0.55:
                logger.warning(
                    "Low printed-OCR confidence (%.1f%%). Image may be handwritten or too small.",
                    avg_confidence * 100,
                )

            logger.info(
                "Printed OCR: %d visual lines, %d text blocks, avg_conf=%.1f%%",
                visual_line_count,
                len(normalized_detections),
                avg_confidence * 100,
            )

            return {
                'success': True,
                'text': final_text,
                'plain_text': plain_text,
                'lines': text_lines,
                'confidence': avg_confidence,
                'mode': 'printed',
                'engine': 'easyocr',
                'line_count': visual_line_count,
            }
        except Exception as e:
            logger.error("EasyOCR recognition failed: %s", e, exc_info=True)
            return {'success': False, 'error': f'EasyOCR error: {str(e)}', 'text': ''}
