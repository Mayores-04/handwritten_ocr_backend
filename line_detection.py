"""
Line detection and layout preservation utilities
"""

import numpy as np


def group_words_into_lines(word_boxes: list, threshold_ratio: float = 0.6) -> list:
    """
    Group detected words into lines based on Y-coordinate overlap
    
    Args:
        word_boxes: List of word boxes with y_center, y_top, y_bottom, x_left, text
        threshold_ratio: Ratio of median height to use as threshold
    
    Returns:
        List of line strings
    """
    if not word_boxes:
        return []
    
    # Sort by Y then X
    word_boxes_sorted = sorted(word_boxes, key=lambda x: (x['y_center'], x['x_left']))
    
    # Calculate median height for threshold
    heights = [wb['height'] for wb in word_boxes_sorted]
    median_height = np.median(heights) if heights else 30
    line_threshold = median_height * threshold_ratio
    
    lines = []
    current_line_words = []
    current_line_y_range = None
    
    for word_box in word_boxes_sorted:
        y_center = word_box['y_center']
        
        if current_line_y_range is None:
            current_line_words = [word_box]
            current_line_y_range = (word_box['y_top'], word_box['y_bottom'])
        else:
            line_y_center = (current_line_y_range[0] + current_line_y_range[1]) / 2
            
            if abs(y_center - line_y_center) <= line_threshold:
                # Same line
                current_line_words.append(word_box)
                current_line_y_range = (
                    min(current_line_y_range[0], word_box['y_top']),
                    max(current_line_y_range[1], word_box['y_bottom'])
                )
            else:
                # New line - save current
                line_text = words_to_line(current_line_words)
                lines.append(line_text)
                
                # Start new line
                current_line_words = [word_box]
                current_line_y_range = (word_box['y_top'], word_box['y_bottom'])
    
    # Don't forget last line
    if current_line_words:
        lines.append(words_to_line(current_line_words))
    
    return lines


def words_to_line(words: list) -> str:
    """Sort words by X position and join into line"""
    sorted_words = sorted(words, key=lambda x: x['x_left'])
    return ' '.join([w['text'] for w in sorted_words])


def simple_line_grouping(word_boxes: list, line_threshold: float = None) -> list:
    """
    Simple line grouping using Y-center distance
    
    Args:
        word_boxes: List with y_center, x_left, text
        line_threshold: Pixel threshold for line break (auto-calculated if None)
    """
    if not word_boxes:
        return []
    
    # Sort by Y then X
    sorted_boxes = sorted(word_boxes, key=lambda x: (x['y_center'], x['x_left']))
    
    # Auto-calculate threshold if not provided
    if line_threshold is None:
        if 'box' in sorted_boxes[0]:
            heights = [abs(wb['box'][2][1] - wb['box'][0][1]) for wb in sorted_boxes]
            line_threshold = np.mean(heights) * 0.8 if heights else 30
        else:
            line_threshold = 30
    
    lines = []
    current_line_words = []
    last_y = -100
    
    for word_box in sorted_boxes:
        y_center = word_box['y_center']
        
        if abs(y_center - last_y) > line_threshold and current_line_words:
            line_text = words_to_line(current_line_words)
            lines.append(line_text)
            current_line_words = []
        
        current_line_words.append(word_box)
        last_y = y_center
    
    if current_line_words:
        lines.append(words_to_line(current_line_words))
    
    return lines
