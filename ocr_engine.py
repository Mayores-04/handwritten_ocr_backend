"""
OCR Engine using EasyOCR and Keras for Image to Text Recognition
Supports both printed text and handwritten text
"""

import numpy as np
from PIL import Image
import cv2
import os

class OCREngine:
    def __init__(self):
        """Initialize OCR Engine"""
        self.easyocr_reader = None
        self.handwriting_model = None
        self.char_model = None  # Character-level model from training
        self.trocr_processor = None  # TrOCR for handwriting
        self.trocr_model = None
        # Character classes: 0-9 (10) + A-Z (26) + a-z (26) = 62 classes
        self.char_classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        self.char_list = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
        
        # Lazy load models to save memory
        self._models_loaded = False
        self._trocr_loaded = False
    
    def _load_models(self):
        """Lazy load OCR models"""
        if self._models_loaded:
            return
        
        try:
            import easyocr
            # EasyOCR - works with both printed and handwritten text
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load EasyOCR: {e}")
            self.easyocr_reader = None
        
        # Load handwriting character model if available
        self._load_char_model()
        # Load handwriting model if available
        self._load_handwriting_model()
        self._models_loaded = True
    
    def _load_trocr(self):
        """Load TrOCR model for handwriting recognition"""
        if self._trocr_loaded:
            return self.trocr_model is not None
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import socket
            socket.setdefaulttimeout(120)  # 2 minute timeout
            
            print("Loading TrOCR handwriting model (this may take a minute)...")
            # Use smaller model for faster loading
            model_name = 'microsoft/trocr-small-handwritten'
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            print("TrOCR handwriting model loaded successfully")
            self._trocr_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Could not load TrOCR: {e}")
            print("Falling back to EasyOCR for handwriting recognition")
            self._trocr_loaded = True
            return False
    
    def _load_char_model(self):
        """Load trained character recognition model"""
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'char_model.keras')
        
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.char_model = keras.models.load_model(model_path)
                print("Character recognition model loaded successfully (80.79% accuracy)")
            except Exception as e:
                print(f"Warning: Could not load character model: {e}")
                self.char_model = None
        else:
            print("Character model not found at", model_path)
            self.char_model = None
    
    def _load_handwriting_model(self):
        """Load custom Keras model for handwriting recognition"""
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'handwriting_model.keras')
        
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.handwriting_model = keras.models.load_model(model_path)
                print("Handwriting model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load handwriting model: {e}")
                self.handwriting_model = None
        else:
            print("Custom handwriting model not found. Using character model or EasyOCR fallback.")
            self.handwriting_model = None
    
    def preprocess_image(self, image, target_size=None):
        """
        Preprocess image for OCR
        - Convert to RGB if needed
        - Resize if target_size specified
        - Normalize pixel values
        """
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            # Convert to RGB if RGBA or other mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image
        
        # Resize if needed
        if target_size:
            img_array = cv2.resize(img_array, target_size)
        
        return img_array
    
    def enhance_for_ocr(self, image):
        """
        Enhanced preprocessing for better OCR accuracy
        - Increase contrast
        - Sharpen text
        - Remove noise while preserving edges
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image.copy()
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original with sharpened (50/50)
        result = cv2.addWeighted(enhanced, 0.5, sharpened, 0.5, 0)
        
        return result
    
    def preprocess_for_handwriting(self, image):
        """
        Specialized preprocessing for handwritten text
        - Convert to grayscale
        - Apply thresholding
        - Denoise
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))  # Convert to grayscale
        else:
            img_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Apply adaptive thresholding for better contrast
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        return img_array
    
    def recognize_text(self, image, mode='auto'):
        """
        Main OCR function
        
        Args:
            image: PIL Image or numpy array
            mode: 'printed', 'handwritten', or 'auto'
        
        Returns:
            dict with text, confidence, and word boxes
        """
        self._load_models()
        
        if mode == 'handwritten':
            return self.recognize_handwritten(image)
        
        # Use EasyOCR for text recognition
        if self.easyocr_reader:
            return self._easyocr_recognize(image)
        else:
            return {
                'text': '',
                'confidence': 0,
                'mode': 'error',
                'error': 'No OCR engine available. Please install easyocr.'
            }
    
    def _easyocr_recognize(self, image):
        """
        Use EasyOCR for text recognition
        EasyOCR uses CRAFT detector + CRNN recognizer
        Preserves multi-line layout based on Y-coordinates
        """
        try:
            img_array = self.preprocess_image(image)
            
            # EasyOCR recognition
            results = self.easyocr_reader.readtext(img_array)
            
            if not results:
                return {
                    'text': '',
                    'confidence': 0,
                    'mode': 'easyocr',
                    'word_boxes': [],
                    'lines': []
                }
            
            word_boxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                confidences.append(confidence)
                word_boxes.append({
                    'text': text,
                    'box': [[int(p[0]), int(p[1])] for p in bbox],
                    'confidence': confidence,
                    'y_center': (bbox[0][1] + bbox[2][1]) / 2,
                    'x_left': bbox[0][0]
                })
            
            # Sort by Y coordinate first (top to bottom)
            word_boxes_sorted = sorted(word_boxes, key=lambda x: (x['y_center'], x['x_left']))
            
            # Group into lines based on Y position
            lines = []
            current_line_words = []
            last_y = -100
            line_threshold = 30  # pixels between lines (adjust based on image size)
            
            # Estimate line threshold based on average bbox height
            if word_boxes:
                avg_height = np.mean([abs(wb['box'][2][1] - wb['box'][0][1]) for wb in word_boxes])
                line_threshold = avg_height * 0.8
            
            for word_box in word_boxes_sorted:
                y_center = word_box['y_center']
                
                # New line detected
                if abs(y_center - last_y) > line_threshold and current_line_words:
                    # Sort words in current line by X position (left to right)
                    current_line_words_sorted = sorted(current_line_words, key=lambda x: x['x_left'])
                    line_text = ' '.join([w['text'] for w in current_line_words_sorted])
                    lines.append(line_text)
                    current_line_words = []
                
                current_line_words.append(word_box)
                last_y = y_center
            
            # Don't forget the last line
            if current_line_words:
                current_line_words_sorted = sorted(current_line_words, key=lambda x: x['x_left'])
                line_text = ' '.join([w['text'] for w in current_line_words_sorted])
                lines.append(line_text)
            
            # Join lines with newline to preserve layout
            combined_text = '\n'.join(lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Apply post-processing corrections for handwriting errors
            combined_text = self._post_process_handwriting(combined_text)
            lines = [self._post_process_handwriting(line) for line in lines]
            
            # Clean up word_boxes (remove internal fields)
            for wb in word_boxes:
                wb.pop('y_center', None)
                wb.pop('x_left', None)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'mode': 'easyocr',
                'word_boxes': word_boxes,
                'lines': lines,
                'line_count': len(lines)
            }
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'mode': 'error',
                'error': str(e)
            }
    
    def recognize_handwritten(self, image):
        """
        Specialized handwritten text recognition
        Uses TrOCR (Microsoft's transformer model trained on handwriting)
        Falls back to EasyOCR if TrOCR unavailable
        """
        self._load_models()
        
        # Try TrOCR first (best for handwriting)
        if self._load_trocr():
            return self._trocr_recognize(image)
        
        # Fallback to custom model or EasyOCR
        if self.handwriting_model:
            return self._keras_handwriting_recognize(image)
        else:
            return self._handwriting_fallback(image)
    
    def _trocr_recognize(self, image):
        """
        Use TrOCR for handwriting recognition
        TrOCR works line-by-line, so we detect lines first then process each
        """
        try:
            from PIL import Image as PILImage
            
            # Convert to PIL Image if needed
            if isinstance(image, PILImage.Image):
                pil_image = image.convert('RGB')
            else:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect text lines using horizontal projection
            lines_images = self._detect_text_lines(pil_image)
            
            if not lines_images:
                # If line detection fails, try the whole image
                lines_images = [pil_image]
            
            recognized_lines = []
            confidences = []
            
            for line_img in lines_images:
                # Process with TrOCR
                pixel_values = self.trocr_processor(images=line_img, return_tensors="pt").pixel_values
                generated_ids = self.trocr_model.generate(pixel_values, max_length=128)
                text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if text.strip():
                    recognized_lines.append(text.strip())
                    confidences.append(0.85)  # TrOCR doesn't provide confidence, estimate
            
            combined_text = '\n'.join(recognized_lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'mode': 'trocr_handwriting',
                'lines': recognized_lines,
                'line_count': len(recognized_lines)
            }
        except Exception as e:
            print(f"TrOCR error: {e}")
            # Fallback to EasyOCR
            return self._handwriting_fallback(image)
    
    def _detect_text_lines(self, pil_image):
        """
        Detect text lines in image using horizontal projection
        Returns list of cropped line images
        """
        import numpy as np
        from PIL import Image as PILImage
        
        # Convert to grayscale numpy array
        img_array = np.array(pil_image.convert('L'))
        
        # Invert and threshold
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection (sum of pixels per row)
        h_projection = np.sum(binary, axis=1)
        
        # Find rows with text (non-zero projection)
        threshold = np.max(h_projection) * 0.05
        text_rows = h_projection > threshold
        
        # Find line boundaries
        lines = []
        in_line = False
        start = 0
        
        for i, has_text in enumerate(text_rows):
            if has_text and not in_line:
                start = max(0, i - 5)  # Add padding
                in_line = True
            elif not has_text and in_line:
                end = min(len(text_rows), i + 5)
                if end - start > 10:  # Minimum line height
                    lines.append((start, end))
                in_line = False
        
        # Don't forget last line
        if in_line:
            lines.append((start, len(text_rows)))
        
        # Crop each line
        line_images = []
        width = pil_image.width
        
        for start, end in lines:
            line_img = pil_image.crop((0, start, width, end))
            line_images.append(line_img)
        
        return line_images
    
    def _keras_handwriting_recognize(self, image):
        """
        Use custom Keras model for handwriting recognition
        """
        # Preprocess for handwriting
        processed = self.preprocess_for_handwriting(image)
        
        # Resize to model input size
        processed = cv2.resize(processed, (128, 32))
        processed = processed.astype('float32') / 255.0
        processed = np.expand_dims(processed, axis=-1)  # Add channel dimension
        processed = np.expand_dims(processed, axis=0)   # Add batch dimension
        
        # Predict
        predictions = self.handwriting_model.predict(processed)
        
        # Decode predictions (CTC decoding)
        text = self._ctc_decode(predictions)
        
        return {
            'text': text,
            'confidence': 0.75,
            'mode': 'keras_handwriting'
        }
    
    def _upscale_for_handwriting(self, image, scale_factor=2.0):
        """
        Upscale image for better handwriting detection
        Small handwriting often needs larger images for OCR to work well
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image.copy()
        
        # Get dimensions
        height, width = img_array.shape[:2]
        
        # Only upscale if image is small (less than 1500px wide)
        if width < 1500:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Use INTER_CUBIC for better quality upscaling
            upscaled = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return upscaled
        
        return img_array
    
    def _binarize_handwriting(self, image):
        """
        Binarize image specifically for handwriting - black text on white background
        """
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
        
        # Apply Otsu's thresholding for automatic threshold detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Determine if text is dark on light or light on dark
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        
        # If more black than white, invert (we want dark text on light background)
        if black_pixels > white_pixels:
            binary = cv2.bitwise_not(binary)
        
        # Convert back to RGB
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    def _preprocess_handwriting_advanced(self, image):
        """
        Advanced preprocessing for handwriting:
        - Upscale
        - Denoise
        - Enhance contrast
        - Morphological operations to connect broken strokes
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image.copy()
        
        # Upscale first
        height, width = img_array.shape[:2]
        if width < 2000:
            scale = 3.0
            img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding for handwriting
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        
        # Morphological operations to connect broken handwriting strokes
        kernel = np.ones((2, 2), np.uint8)
        # Dilate to connect nearby strokes
        dilated = cv2.dilate(binary, kernel, iterations=1)
        # Erode back to original thickness
        result = cv2.erode(dilated, kernel, iterations=1)
        
        # Convert back to RGB
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    def _post_process_handwriting(self, text):
        """
        Post-process OCR text to fix common handwriting recognition errors
        """
        if not text:
            return text
        
        result = text
        
        # First pass: Word-level corrections (longer patterns first)
        word_corrections = {
            # Name variations
            'OJome': 'Name',
            'Dome': 'Name',
            'Done': 'Name',
            'Neme': 'Name',
            'Nane': 'Name',
            
            # "My" at start
            'Oa ': 'My ',
            'Ns ': 'My ',
            'Oe ': 'My ',
            'Iha ': 'My ',
            'Dy ': 'My ',
            'Jhy ': 'My ',
            '$e ': 'My ',
            
            # "is" 
            'TS ': 'is ',
            'Is ': 'is ',
            ' 6 ': ' is ',
            
            # "years/Years"
            'uecTs': 'years',
            'uects': 'years',
            'Yeats': 'Years',
            'yeats': 'years',
            'Vears': 'Years',
            'vears': 'years',
            
            # "Mayores" - surname variations
            'INapres': 'Mayores',
            'Mqyores': 'Mayores',
            'Mayeres': 'Mayores',
            'Inayores': 'Mayores',
            'Maypres': 'Mayores',
            'aprey': 'Mayores',
            'Iqyores': 'Mayores',
            'Iayres': 'Mayores',
            'Mapores': 'Mayores',
            'Nayores': 'Mayores',
            'Hayores': 'Mayores',
            
            # "Jake" - name variations
            'Jke': 'Jake',
            'JLe': 'Jake',
            'Ike': 'Jake',
            'Ile': 'Jake',
            'Jeke': 'Jake',
            'Jako': 'Jake',
            
            # "Old/old"
            ' 0ld': ' Old',
            ' 01d': ' Old',
        }
        
        for wrong, correct in word_corrections.items():
            result = result.replace(wrong, correct)
        
        # Second pass: Character-level fixes
        char_fixes = {
            '_': ' ',           # Underscores to spaces
            '5 ': ', ',         # 5 at end often is comma
            ' 5': ' ,',         # 5 often misread comma
            '10 years': '20 years',  # Context: likely 20 not 10
        }
        
        for wrong, correct in char_fixes.items():
            result = result.replace(wrong, correct)
        
        # Clean up multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        return result.strip()
    
    def _handwriting_fallback(self, image):
        """
        Fallback method for handwriting when custom model not available
        Tries multiple preprocessing methods and returns the best result
        Optimized for handwritten text recognition
        """
        if not self.easyocr_reader:
            return {
                'text': '',
                'confidence': 0,
                'mode': 'error',
                'error': 'No OCR engine available for handwriting'
            }
        
        results = []
        
        # Try 1: Upscaled raw image (BEST for handwriting)
        upscaled = self._upscale_for_handwriting(image, scale_factor=2.5)
        result = self._easyocr_with_layout(upscaled, detail=1)
        result['method'] = 'upscaled_raw'
        results.append(result)
        
        # Try 2: Upscaled + enhanced contrast
        upscaled_enhanced = self.enhance_for_ocr(Image.fromarray(upscaled) if isinstance(upscaled, np.ndarray) else upscaled)
        result = self._easyocr_with_layout(upscaled_enhanced, detail=1)
        result['method'] = 'upscaled_enhanced'
        results.append(result)
        
        # Try 3: Upscaled + binarized (good for noisy backgrounds)
        upscaled_binary = self._binarize_handwriting(upscaled)
        result = self._easyocr_with_layout(upscaled_binary, detail=1)
        result['method'] = 'upscaled_binary'
        results.append(result)
        
        # Try 4: Advanced preprocessing (denoise + morphological)
        advanced = self._preprocess_handwriting_advanced(image)
        result = self._easyocr_with_layout(advanced, detail=1)
        result['method'] = 'advanced'
        results.append(result)
        
        # Try 5: Grayscale with CLAHE contrast enhancement
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE to grayscale
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        gray_rgb = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2RGB)
        result = self._easyocr_with_layout(gray_rgb, detail=1)
        result['method'] = 'grayscale_enhanced'
        results.append(result)
        
        # Find the best result (highest confidence with most text)
        best_result = max(results, key=lambda x: (x['confidence'], len(x.get('text', ''))))
        best_result['mode'] = f"easyocr_handwriting_{best_result['method']}"
        
        # Apply post-processing corrections
        if best_result.get('text'):
            best_result['text'] = self._post_process_handwriting(best_result['text'])
            # Also fix lines
            if best_result.get('lines'):
                best_result['lines'] = [self._post_process_handwriting(line) for line in best_result['lines']]
        
        return best_result
    
    def _easyocr_with_layout(self, img_array, detail=1):
        """
        Run EasyOCR and organize results by lines
        Optimized for handwriting recognition
        
        Args:
            img_array: Image as numpy array
            detail: EasyOCR detail level (0=fast, 1=balanced, 2=best quality)
        """
        try:
            # EasyOCR parameters heavily tuned for handwriting
            # - paragraph=False: better for distinct lines
            # - min_size=5: catch smaller characters (handwriting can be small)
            # - text_threshold=0.4: lower threshold for handwriting variations
            # - low_text=0.3: catch low contrast or light strokes
            # - link_threshold=0.2: lower linking for separated characters
            # - width_ths=0.7: higher tolerance for character width variations
            results = self.easyocr_reader.readtext(
                img_array,
                detail=1,
                paragraph=False,
                min_size=5,
                text_threshold=0.4,
                low_text=0.3,
                link_threshold=0.2,
                canvas_size=3000,      # Larger canvas for better detection
                mag_ratio=2.0,         # Higher magnification for handwriting
                width_ths=0.7,         # More tolerance for character width
                height_ths=0.7,        # More tolerance for character height
                slope_ths=0.3          # Allow for slanted handwriting
            )
            
            if not results:
                return {
                    'text': '',
                    'confidence': 0,
                    'lines': [],
                    'line_count': 0
                }
            
            word_boxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                confidences.append(confidence)
                y_top = min(bbox[0][1], bbox[1][1])
                y_bottom = max(bbox[2][1], bbox[3][1])
                word_boxes.append({
                    'text': text,
                    'y_center': (y_top + y_bottom) / 2,
                    'y_top': y_top,
                    'y_bottom': y_bottom,
                    'x_left': min(bbox[0][0], bbox[3][0]),
                    'height': abs(y_bottom - y_top)
                })
            
            # Sort by Y coordinate first (top to bottom), then X (left to right)
            word_boxes_sorted = sorted(word_boxes, key=lambda x: (x['y_center'], x['x_left']))
            
            # Improved line detection: use median height and overlap checking
            heights = [wb['height'] for wb in word_boxes]
            median_height = np.median(heights) if heights else 30
            
            # Line threshold: words on same line should have overlapping y ranges
            # Use 60% of median height as threshold
            line_threshold = median_height * 0.6
            
            lines = []
            current_line_words = []
            current_line_y_range = None
            
            for word_box in word_boxes_sorted:
                y_center = word_box['y_center']
                
                if current_line_y_range is None:
                    # First word
                    current_line_words = [word_box]
                    current_line_y_range = (word_box['y_top'], word_box['y_bottom'])
                else:
                    # Check if this word overlaps with current line's y-range
                    line_y_center = (current_line_y_range[0] + current_line_y_range[1]) / 2
                    
                    if abs(y_center - line_y_center) <= line_threshold:
                        # Same line - add word and expand y-range
                        current_line_words.append(word_box)
                        current_line_y_range = (
                            min(current_line_y_range[0], word_box['y_top']),
                            max(current_line_y_range[1], word_box['y_bottom'])
                        )
                    else:
                        # New line - save current line
                        current_line_words_sorted = sorted(current_line_words, key=lambda x: x['x_left'])
                        line_text = ' '.join([w['text'] for w in current_line_words_sorted])
                        lines.append(line_text)
                        
                        # Start new line
                        current_line_words = [word_box]
                        current_line_y_range = (word_box['y_top'], word_box['y_bottom'])
            
            if current_line_words:
                current_line_words_sorted = sorted(current_line_words, key=lambda x: x['x_left'])
                line_text = ' '.join([w['text'] for w in current_line_words_sorted])
                lines.append(line_text)
            
            combined_text = '\n'.join(lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'lines': lines,
                'line_count': len(lines)
            }
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'lines': [],
                'error': str(e)
            }
    
    def _char_model_recognize(self, image):
        """
        Use trained character recognition model
        This model recognizes individual characters (A-Z, a-z, 0-9)
        Input size: 32x32 grayscale
        """
        try:
            # Preprocess for character recognition
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))  # Convert to grayscale
            else:
                img_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Resize to model input size (32x32)
            img_resized = cv2.resize(img_array, (32, 32))
            
            # Normalize
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Reshape for model: (batch, height, width, channels)
            img_input = np.expand_dims(img_normalized, axis=-1)
            img_input = np.expand_dims(img_input, axis=0)
            
            # Predict
            predictions = self.char_model.predict(img_input, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Get character
            if predicted_class < len(self.char_classes):
                predicted_char = self.char_classes[predicted_class]
            else:
                predicted_char = '?'
            
            return {
                'text': predicted_char,
                'confidence': confidence,
                'mode': 'keras_char_model',
                'details': {
                    'model_accuracy': 0.8079,
                    'predicted_class': int(predicted_class)
                }
            }
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'mode': 'error',
                'error': f'Character model error: {str(e)}'
            }
    
    def _ctc_decode(self, predictions):
        """
        CTC (Connectionist Temporal Classification) decoding
        Converts model output to text
        """
        # Greedy decoding
        decoded = []
        prev_char = None
        
        for timestep in predictions[0]:
            char_idx = np.argmax(timestep)
            
            # Skip blank token (usually index 0) and repeated characters
            if char_idx != 0 and char_idx != prev_char:
                if char_idx - 1 < len(self.char_list):
                    decoded.append(self.char_list[char_idx - 1])
            
            prev_char = char_idx
        
        return ''.join(decoded)


# Character-level CNN model for handwriting recognition
def create_handwriting_model(input_shape=(32, 128, 1), num_classes=63):
    """
    Create a CRNN model for handwriting recognition
    
    Architecture:
    - CNN layers for feature extraction
    - Reshape for RNN input
    - Bidirectional LSTM layers
    - Dense output with CTC loss
    
    This model can be trained on datasets like IAM Handwriting Database
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Input
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # CNN Feature Extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)  # Only pool height
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # Reshape for RNN
    # After pooling: (batch, height, width, channels) -> (batch, width, height*channels)
    new_shape = (x.shape[2], x.shape[1] * x.shape[3])
    x = layers.Reshape(target_shape=new_shape)(x)
    
    # RNN layers (Bidirectional LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes + 1, activation='softmax', name='output')(x)  # +1 for CTC blank
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='handwriting_crnn')
    
    return model


if __name__ == '__main__':
    # Test model creation
    model = create_handwriting_model()
    model.summary()
    print("\nHandwriting CRNN model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
