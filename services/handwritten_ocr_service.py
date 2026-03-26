"""Handwritten OCR service using Keras character model."""

import logging
from typing import Any, List

import numpy as np

from config import CHAR_CLASSES
from preprocessing.image_processors import preprocess_image
from preprocessing.image_utils import otsu_threshold, resize_char

logger = logging.getLogger(__name__)


class HandwrittenOCRService:
    """Handwritten text OCR pipeline (character segmentation + classification)."""

    def __init__(self, keras_confidence_threshold: float = 0.20):
        self.keras_confidence_threshold = keras_confidence_threshold

    def recognize(self, image: Any, char_model: Any) -> dict[str, Any]:
        try:
            if not char_model:
                return {
                    'success': False,
                    'error': 'Character model not loaded',
                    'text': '',
                    'engine': 'keras',
                }

            pil_image = preprocess_image(image)
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                img_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
            img_array = img_array.astype('uint8')

            thresh = otsu_threshold(img_array)
            binary = (img_array <= thresh).astype('uint8')

            try:
                from scipy.ndimage import binary_dilation, binary_erosion

                selem = np.ones((2, 2), dtype=bool)
                binary = binary_dilation(binary, structure=selem, iterations=1).astype('uint8')
                binary = binary_erosion(binary, structure=selem, iterations=1).astype('uint8')
            except ImportError:
                logger.debug("scipy not available, skipping morphological operations")

            lines = self._segment_lines(binary, img_array)

            all_lines_text = []
            all_confs = []

            for line_bin, line_gray in lines:
                char_images = self._segment_chars_cc(line_bin, line_gray)
                if not char_images:
                    continue

                line_text = ""
                line_confs = []
                prev_x = 0

                for char_gray_img, char_x in char_images:
                    gap = char_x - prev_x
                    if prev_x > 0 and gap > char_gray_img.shape[1] * 1.2:
                        line_text += ' '
                    prev_x = char_x + char_gray_img.shape[1]

                    char_resized = resize_char(char_gray_img, 28, 28)
                    char_data = np.expand_dims(char_resized, axis=-1).astype('float32') / 255.0
                    preds = char_model.predict(np.expand_dims(char_data, 0), verbose=0)

                    top3_idx = np.argsort(preds[0])[-3:][::-1]
                    top3_conf = preds[0][top3_idx]

                    class_idx = int(top3_idx[0])
                    confidence = float(top3_conf[0])

                    if confidence >= self.keras_confidence_threshold and class_idx < len(CHAR_CLASSES):
                        line_text += CHAR_CLASSES[class_idx]
                        line_confs.append(confidence)
                    elif len(top3_idx) > 1 and top3_conf[1] >= self.keras_confidence_threshold * 0.8:
                        second_class_idx = int(top3_idx[1])
                        if second_class_idx < len(CHAR_CLASSES):
                            line_text += CHAR_CLASSES[second_class_idx]
                            line_confs.append(float(top3_conf[1]))

                if line_text.strip():
                    all_lines_text.append(line_text)
                    all_confs.append(float(np.mean(line_confs)) if line_confs else 0.0)

            if not all_lines_text:
                return {
                    'success': False,
                    'error': 'No text recognized',
                    'text': '',
                    'engine': 'keras',
                }

            combined_text = '\n'.join(all_lines_text)
            plain_text = ' '.join(all_lines_text)
            avg_conf = float(np.mean(all_confs)) if all_confs else 0.0

            return {
                'success': True,
                'text': combined_text,
                'plain_text': plain_text,
                'lines': all_lines_text,
                'confidence': avg_conf,
                'mode': 'handwritten',
                'engine': 'keras_char_model',
                'line_count': len(all_lines_text),
            }
        except Exception as e:
            logger.error("Handwritten recognition failed: %s", e, exc_info=True)
            return {
                'success': False,
                'error': f'Handwritten error: {e}',
                'text': '',
                'engine': 'keras',
            }

    def _segment_lines(self, binary: np.ndarray, gray: np.ndarray) -> List[tuple]:
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

        # NEW: Group fragments by VERTICAL POSITION (not just small gaps)
        # Single-line handwritten text has all fragments at same vertical level
        fragments = [(s, e) for s, e in zip(starts, ends)]
        heights = [e - s for s, e in fragments]
        mean_band_h = np.mean(heights)
        
        # Merge fragments that overlap or are at same vertical baseline
        merged = []
        used = [False] * len(fragments)
        
        for i in range(len(fragments)):
            if used[i]:
                continue
            
            start, end = fragments[i]
            used[i] = True
            
            # Check if other fragments are at same vertical position
            for j in range(i + 1, len(fragments)):
                if used[j]:
                    continue
                
                s2, e2 = fragments[j]
                
                # Overlapping fragments = definitely same line
                if max(start, s2) < min(end, e2):
                    start = min(start, s2)
                    end = max(end, e2)
                    used[j] = True
                # Close fragments on same baseline = same line
                elif abs(s2 - end) <= mean_band_h * 0.5 or abs(start - e2) <= mean_band_h * 0.5:
                    start = min(start, s2)
                    end = max(end, e2)
                    used[j] = True
            
            merged.append((start, end))

        pad = max(int(mean_band_h * 0.10), 2)
        lines = []
        for r0, r1 in zip([s for s, e in merged], [e for s, e in merged]):
            r0p = max(0, r0 - pad)
            r1p = min(h, r1 + pad)

            line_bin = binary[r0p:r1p, :]
            line_gray = gray[r0p:r1p, :]

            if line_bin.shape[0] < 3 or line_bin.sum() == 0:
                continue

            lines.append((line_bin, line_gray))

        return lines if lines else [(binary, gray)]

    def _segment_chars_cc(self, line_bin: np.ndarray, line_gray: np.ndarray) -> List[tuple]:
        h, w = line_bin.shape
        if h < 3 or w < 3:
            return []

        try:
            from scipy.ndimage import label

            labeled, num_features = label(line_bin)
        except ImportError:
            return self._segment_chars_vproj(line_bin, line_gray)

        if num_features == 0:
            return []

        chars = []
        for comp_id in range(1, num_features + 1):
            mask = labeled == comp_id
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue

            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1

            comp_h = r1 - r0
            comp_w = c1 - c0

            ink_area = mask.sum()
            bbox_area = comp_h * comp_w
            ink_ratio = ink_area / bbox_area if bbox_area > 0 else 0

            min_h = max(2, h * 0.05)
            min_w = 2
            max_w = w * 0.6
            max_h = h * 0.95

            if (
                comp_h < min_h
                or comp_w < min_w
                or comp_w > max_w
                or comp_h > max_h
                or ink_ratio < 0.15
                or ink_ratio > 0.98
            ):
                continue

            char_crop = line_gray[r0:r1, c0:c1]
            chars.append((char_crop, c0))

        chars.sort(key=lambda x: x[1])
        return chars

    def _segment_chars_vproj(self, line_bin: np.ndarray, line_gray: np.ndarray) -> List[tuple]:
        _, w = line_bin.shape
        v_proj = line_bin.sum(axis=0).astype(float)
        max_proj = v_proj.max()
        if max_proj == 0:
            return []

        char_threshold = max(max_proj * 0.05, 1.0)
        text_cols = v_proj > char_threshold

        diffs = np.diff(text_cols.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        chars = []
        for start, end in zip(starts, ends):
            if end - start < 2:
                continue
            c0 = max(0, start - 1)
            c1 = min(w, end + 1)
            char_crop = line_gray[:, c0:c1]
            chars.append((char_crop, c0))

        return chars
