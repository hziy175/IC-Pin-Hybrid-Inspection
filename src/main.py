from recipe import RECIPE, BODY_THRESHOLD, MIN_BODY_AREA_RATIO, MAX_BODY_AREA_RATIO, BODY_CENTER_MIN_RATIO, BODY_CENTER_MAX_RATIO, MIN_BODY_RECTANGULARITY, PIN_THRESHOLD, PIN_BAND_RATIO, X_MARGIN_RATIO, Y_MARGIN_RATIO, MIN_PIN_AREA, MAX_PIN_AREA, MIN_PIN_ASPECT, MAX_PIN_ASPECT, EXPECTED_PIN_COUNT, MATCH_TOLERANCE_RATIO, MISSING_GAP_ERROR_LIMIT, PITCH_ERROR_LIMIT, POSITION_SHIFT_LIMIT_PITCH_RATIO, WIDTH_ERROR_LIMIT, HEIGHT_ERROR_LIMIT, AREA_ERROR_LIMIT, ASPECT_RATIO_ERROR_LIMIT, FILL_RATIO_ERROR_LIMIT, GEOMETRIC_REFERENCE_IMAGE
from pathlib import Path
from statistics import median
import cv2
import numpy as np
import onnxruntime as ort
import config
from utils import read_image, save_image
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_IMAGE = ROOT_DIR / GEOMETRIC_REFERENCE_IMAGE
TEST_IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else 'test_chip.jpg'
TEST_IMAGE = ROOT_DIR / 'data' / 'input' / TEST_IMAGE_NAME
DESKEWED_TEST_IMAGE = config.DESKEWED_DIR / f'{Path(TEST_IMAGE_NAME).stem}_deskewed.jpg'
OUTPUT_IMAGE = config.RESULT_DIR / f'{Path(TEST_IMAGE_NAME).stem}_result.jpg'
ONNX_MODEL_PATH = config.MODEL_PATH
CNN_DIAGNOSTIC_IMAGE = config.CNN_DIAGNOSTIC_DIR / f'{Path(TEST_IMAGE_NAME).stem}_cnn.jpg'
CNN_PATCH_WIDTH = int(RECIPE['cnn']['patch_width'])
CNN_PATCH_HEIGHT = int(RECIPE['cnn']['patch_height'])
CNN_DEFECT_THRESHOLD = float(RECIPE['cnn']['defect_threshold'])
CNN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CNN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def find_ic_body(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BODY_THRESHOLD, 255, cv2.THRESH_BINARY)
    img_h, img_w = img.shape[:2]
    image_area = img_h * img_w
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        area_ratio = area / image_area
        x, y, bw, bh = cv2.boundingRect(contour)
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        rectangularity = area / (rw * rh)
        if area_ratio < MIN_BODY_AREA_RATIO or area_ratio > MAX_BODY_AREA_RATIO:
            continue
        if cx < BODY_CENTER_MIN_RATIO * img_w or cx > BODY_CENTER_MAX_RATIO * img_w:
            continue
        if cy < BODY_CENTER_MIN_RATIO * img_h or cy > BODY_CENTER_MAX_RATIO * img_h:
            continue
        if rectangularity < MIN_BODY_RECTANGULARITY:
            continue
        candidates.append((area, rectangularity, contour))
    if not candidates:
        raise RuntimeError('No valid IC body candidate found.')
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][2]

def deskew_image(img):
    body_contour = find_ic_body(img)
    rect = cv2.minAreaRect(body_contour)
    center = rect[0]
    rect_w, rect_h = rect[1]
    raw_angle = rect[2]
    normalized_angle = raw_angle
    if rect_w < rect_h:
        normalized_angle += 90.0
    img_h, img_w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, normalized_angle, 1.0)
    corrected_img = cv2.warpAffine(img, rotation_matrix, (img_w, img_h))
    return {'image': corrected_img, 'raw_angle': raw_angle, 'normalized_angle': normalized_angle}

def calculate_pin_rois(img, body_contour):
    img_h, img_w = img.shape[:2]
    x, y, body_w, body_h = cv2.boundingRect(body_contour)
    pin_band_h = int(round(body_h * PIN_BAND_RATIO))
    x_margin = int(round(body_w * X_MARGIN_RATIO))
    y_margin = int(round(body_h * Y_MARGIN_RATIO))
    roi_x1 = max(0, x - x_margin)
    roi_x2 = min(img_w, x + body_w + x_margin)
    top_y1 = max(0, y - y_margin)
    top_y2 = min(img_h, y + pin_band_h)
    bottom_y1 = max(0, y + body_h - pin_band_h)
    bottom_y2 = min(img_h, y + body_h + y_margin)
    top_roi = img[top_y1:top_y2, roi_x1:roi_x2]
    bottom_roi = img[bottom_y1:bottom_y2, roi_x1:roi_x2]
    if top_roi.size == 0:
        raise RuntimeError('Top ROI is empty.')
    if bottom_roi.size == 0:
        raise RuntimeError('Bottom ROI is empty.')
    top_box = (roi_x1, top_y1, roi_x2, top_y2)
    bottom_box = (roi_x1, bottom_y1, roi_x2, bottom_y2)
    body_box = (x, y, body_w, body_h)
    return (top_roi, bottom_roi, top_box, bottom_box, body_box)

def detect_pins(roi, offset_x, offset_y):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, PIN_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    raw_pins = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]
        if area < MIN_PIN_AREA or area > MAX_PIN_AREA:
            continue
        aspect_ratio = bh / bw
        if aspect_ratio < MIN_PIN_ASPECT or aspect_ratio > MAX_PIN_ASPECT:
            continue
        global_x = offset_x + x
        global_y = offset_y + y
        global_cx = offset_x + cx
        global_cy = offset_y + cy
        raw_pins.append((label, global_x, global_y, bw, bh, area, global_cx, global_cy))
    raw_pins.sort(key=lambda item: item[6])
    return (raw_pins, binary)

def process_image(img):
    body_contour = find_ic_body(img)
    top_roi, bottom_roi, top_box, bottom_box, body_box = calculate_pin_rois(img, body_contour)
    top_x1, top_y1, _, _ = top_box
    bottom_x1, bottom_y1, _, _ = bottom_box
    top_raw_pins, top_binary = detect_pins(top_roi, top_x1, top_y1)
    bottom_raw_pins, bottom_binary = detect_pins(bottom_roi, bottom_x1, bottom_y1)
    return {'body_box': body_box, 'top_box': top_box, 'bottom_box': bottom_box, 'top_raw_pins': top_raw_pins, 'bottom_raw_pins': bottom_raw_pins, 'top_binary': top_binary, 'bottom_binary': bottom_binary}

def build_pin_records(raw_pins, body_box, prefix=None):
    body_x, body_y, body_w, body_h = body_box
    pin_records = []
    for index, pin in enumerate(raw_pins, start=1):
        label, x, y, bw, bh, area, cx, cy = pin
        if prefix is None:
            pin_id = None
        else:
            pin_id = f'{prefix}{index:02d}'
        relative_x = (cx - body_x) / body_w
        relative_y = (cy - body_y) / body_h
        pin_records.append({'pin_id': pin_id, 'label': label, 'x': x, 'y': y, 'width': bw, 'height': bh, 'area': area, 'cx': cx, 'cy': cy, 'relative_x': relative_x, 'relative_y': relative_y})
    return pin_records

def calculate_reference_pitch_ratio(reference_pins):
    pitch_ratios = []
    for i in range(len(reference_pins) - 1):
        left_pin = reference_pins[i]
        right_pin = reference_pins[i + 1]
        pitch_ratio = right_pin['relative_x'] - left_pin['relative_x']
        pitch_ratios.append(pitch_ratio)
    if not pitch_ratios:
        raise RuntimeError('Not enough reference pins to calculate pitch.')
    return median(pitch_ratios)

def match_detected_pins_to_reference(reference_pins, detected_pins):
    reference_pitch_ratio = calculate_reference_pitch_ratio(reference_pins)
    match_tolerance_ratio = reference_pitch_ratio * MATCH_TOLERANCE_RATIO
    unmatched_detected_indices = set(range(len(detected_pins)))
    reference_to_detected_pin = {}
    for reference_pin in reference_pins:
        pin_id = reference_pin['pin_id']
        if not unmatched_detected_indices:
            reference_to_detected_pin[pin_id] = None
            continue
        nearest_detected_index = min(unmatched_detected_indices, key=lambda index: abs(detected_pins[index]['relative_x'] - reference_pin['relative_x']))
        nearest_detected_pin = detected_pins[nearest_detected_index]
        position_difference_ratio = abs(nearest_detected_pin['relative_x'] - reference_pin['relative_x'])
        if position_difference_ratio <= match_tolerance_ratio:
            matched_pin = nearest_detected_pin.copy()
            matched_pin['pin_id'] = pin_id
            reference_to_detected_pin[pin_id] = matched_pin
            unmatched_detected_indices.remove(nearest_detected_index)
        else:
            reference_to_detected_pin[pin_id] = None
    missing_pin_ids = []
    for pin_id, detected_pin in reference_to_detected_pin.items():
        if detected_pin is None:
            missing_pin_ids.append(pin_id)
    unmatched_pins = [detected_pins[index] for index in sorted(unmatched_detected_indices)]
    return {'reference_to_detected_pin': reference_to_detected_pin, 'missing_pin_ids': missing_pin_ids, 'unmatched_pins': unmatched_pins, 'reference_pitch_ratio': reference_pitch_ratio, 'match_tolerance_ratio': match_tolerance_ratio}

def check_missing_gap_evidence(reference_pins, match_result, test_body_box):
    _, _, body_w, _ = test_body_box
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    evidence_list = []
    index = 0
    while index < len(reference_pins):
        current_id = reference_pins[index]['pin_id']
        if reference_to_detected_pin[current_id] is not None:
            index += 1
            continue
        missing_start = index
        while index < len(reference_pins):
            pin_id = reference_pins[index]['pin_id']
            if reference_to_detected_pin[pin_id] is not None:
                break
            index += 1
        missing_end = index - 1
        missing_ids = [reference_pins[i]['pin_id'] for i in range(missing_start, missing_end + 1)]
        left_index = missing_start - 1
        right_index = index
        if left_index < 0 or right_index >= len(reference_pins):
            evidence_list.append({'missing_pin_ids': missing_ids, 'has_two_sided_evidence': False})
            continue
        left_reference_pin = reference_pins[left_index]
        right_reference_pin = reference_pins[right_index]
        left_pin_id = left_reference_pin['pin_id']
        right_pin_id = right_reference_pin['pin_id']
        left_detected_pin = reference_to_detected_pin[left_pin_id]
        right_detected_pin = reference_to_detected_pin[right_pin_id]
        if left_detected_pin is None or right_detected_pin is None:
            evidence_list.append({'missing_pin_ids': missing_ids, 'has_two_sided_evidence': False})
            continue
        expected_gap_ratio = right_reference_pin['relative_x'] - left_reference_pin['relative_x']
        expected_gap_px = expected_gap_ratio * body_w
        detected_gap_px = right_detected_pin['cx'] - left_detected_pin['cx']
        gap_error_ratio = abs(detected_gap_px - expected_gap_px) / expected_gap_px
        is_consistent = gap_error_ratio <= MISSING_GAP_ERROR_LIMIT
        evidence_list.append({'missing_pin_ids': missing_ids, 'has_two_sided_evidence': True, 'left_pin_id': left_pin_id, 'right_pin_id': right_pin_id, 'detected_gap_px': detected_gap_px, 'expected_gap_px': expected_gap_px, 'error_ratio': gap_error_ratio, 'is_consistent': is_consistent})
    return {'evidence': evidence_list}

def check_pitch_abnormalities(reference_pins, match_result, test_body_box):
    _, _, body_w, _ = test_body_box
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    abnormal_pairs = []
    checked_pair_count = 0
    for i in range(len(reference_pins) - 1):
        left_reference_pin = reference_pins[i]
        right_reference_pin = reference_pins[i + 1]
        left_pin_id = left_reference_pin['pin_id']
        right_pin_id = right_reference_pin['pin_id']
        left_detected_pin = reference_to_detected_pin[left_pin_id]
        right_detected_pin = reference_to_detected_pin[right_pin_id]
        if left_detected_pin is None or right_detected_pin is None:
            continue
        checked_pair_count += 1
        expected_pitch_ratio = right_reference_pin['relative_x'] - left_reference_pin['relative_x']
        expected_pitch_px = expected_pitch_ratio * body_w
        detected_pitch_px = right_detected_pin['cx'] - left_detected_pin['cx']
        pitch_error_ratio = abs(detected_pitch_px - expected_pitch_px) / expected_pitch_px
        if pitch_error_ratio > PITCH_ERROR_LIMIT:
            abnormal_pairs.append({'left_pin_id': left_pin_id, 'right_pin_id': right_pin_id, 'detected_pitch_px': detected_pitch_px, 'expected_pitch_px': expected_pitch_px, 'error_ratio': pitch_error_ratio, 'left_cx': left_detected_pin['cx'], 'left_cy': left_detected_pin['cy'], 'right_cx': right_detected_pin['cx'], 'right_cy': right_detected_pin['cy']})
    return {'checked_pair_count': checked_pair_count, 'abnormal_pairs': abnormal_pairs}

def check_position_shifts(reference_pins, match_result, test_body_box):
    _ = test_body_box
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    matched_pairs = []
    for reference_pin in reference_pins:
        pin_id = reference_pin['pin_id']
        detected_pin = reference_to_detected_pin[pin_id]
        if detected_pin is None:
            continue
        matched_pairs.append({'reference_pin': reference_pin, 'detected_pin': detected_pin})
    if len(matched_pairs) < 2:
        return {'reference_pitch_px': 0.0, 'detected_pitch_px': 0.0, 'scale_x': 1.0, 'offset_x': 0.0, 'offset_y': 0.0, 'shift_limit_px': 0.0, 'shifted_pins': []}
    reference_pitch_values = []
    detected_pitch_values = []
    for i in range(len(reference_pins) - 1):
        left_reference_pin = reference_pins[i]
        right_reference_pin = reference_pins[i + 1]
        left_pin_id = left_reference_pin['pin_id']
        right_pin_id = right_reference_pin['pin_id']
        left_detected_pin = reference_to_detected_pin[left_pin_id]
        right_detected_pin = reference_to_detected_pin[right_pin_id]
        if left_detected_pin is None or right_detected_pin is None:
            continue
        reference_pitch = right_reference_pin['cx'] - left_reference_pin['cx']
        detected_pitch = right_detected_pin['cx'] - left_detected_pin['cx']
        reference_pitch_values.append(reference_pitch)
        detected_pitch_values.append(detected_pitch)
    if not reference_pitch_values:
        return {'reference_pitch_px': 0.0, 'detected_pitch_px': 0.0, 'scale_x': 1.0, 'offset_x': 0.0, 'offset_y': 0.0, 'shift_limit_px': 0.0, 'shifted_pins': []}
    reference_pitch_px = median(reference_pitch_values)
    detected_pitch_px = median(detected_pitch_values)
    scale_x = detected_pitch_px / reference_pitch_px
    x_offset_values = []
    y_offset_values = []
    for pair in matched_pairs:
        reference_pin = pair['reference_pin']
        detected_pin = pair['detected_pin']
        x_offset = detected_pin['cx'] - reference_pin['cx'] * scale_x
        y_offset = detected_pin['cy'] - reference_pin['cy']
        x_offset_values.append(x_offset)
        y_offset_values.append(y_offset)
    offset_x = median(x_offset_values)
    offset_y = median(y_offset_values)
    shift_limit_px = detected_pitch_px * POSITION_SHIFT_LIMIT_PITCH_RATIO
    shifted_pins = []
    for pair in matched_pairs:
        reference_pin = pair['reference_pin']
        detected_pin = pair['detected_pin']
        pin_id = reference_pin['pin_id']
        expected_cx = reference_pin['cx'] * scale_x + offset_x
        expected_cy = reference_pin['cy'] + offset_y
        shift_x_px = detected_pin['cx'] - expected_cx
        shift_y_px = detected_pin['cy'] - expected_cy
        is_shifted = abs(shift_x_px) > shift_limit_px or abs(shift_y_px) > shift_limit_px
        if is_shifted:
            shifted_pins.append({'pin_id': pin_id, 'shift_x_px': shift_x_px, 'shift_y_px': shift_y_px, 'shift_limit_px': shift_limit_px, 'expected_cx': expected_cx, 'expected_cy': expected_cy, 'detected_cx': detected_pin['cx'], 'detected_cy': detected_pin['cy'], 'x': detected_pin['x'], 'y': detected_pin['y'], 'width': detected_pin['width'], 'height': detected_pin['height']})
    return {'reference_pitch_px': reference_pitch_px, 'detected_pitch_px': detected_pitch_px, 'scale_x': scale_x, 'offset_x': offset_x, 'offset_y': offset_y, 'shift_limit_px': shift_limit_px, 'shifted_pins': shifted_pins}

def check_pin_size_abnormalities(reference_pins, match_result):
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    abnormal_pins = []
    checked_pin_count = 0
    for reference_pin in reference_pins:
        pin_id = reference_pin['pin_id']
        detected_pin = reference_to_detected_pin[pin_id]
        if detected_pin is None:
            continue
        checked_pin_count += 1
        reference_width = reference_pin['width']
        detected_width = detected_pin['width']
        width_error_ratio = abs(detected_width - reference_width) / reference_width
        is_width_abnormal = width_error_ratio > WIDTH_ERROR_LIMIT
        reference_height = reference_pin['height']
        detected_height = detected_pin['height']
        height_error_ratio = abs(detected_height - reference_height) / reference_height
        is_height_abnormal = height_error_ratio > HEIGHT_ERROR_LIMIT
        reference_area = reference_pin['area']
        detected_area = detected_pin['area']
        area_error_ratio = abs(detected_area - reference_area) / reference_area
        is_area_abnormal = area_error_ratio > AREA_ERROR_LIMIT
        is_size_abnormal = is_width_abnormal or is_height_abnormal or is_area_abnormal
        if not is_size_abnormal:
            continue
        abnormal_pins.append({'pin_id': pin_id, 'reference_width': reference_width, 'detected_width': detected_width, 'width_error_ratio': width_error_ratio, 'is_width_abnormal': is_width_abnormal, 'reference_height': reference_height, 'detected_height': detected_height, 'height_error_ratio': height_error_ratio, 'is_height_abnormal': is_height_abnormal, 'reference_area': reference_area, 'detected_area': detected_area, 'area_error_ratio': area_error_ratio, 'is_area_abnormal': is_area_abnormal, 'x': detected_pin['x'], 'y': detected_pin['y'], 'width': detected_pin['width'], 'height': detected_pin['height']})
    return {'checked_pin_count': checked_pin_count, 'abnormal_pins': abnormal_pins}

def check_shape_candidates(reference_pins, match_result):
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    candidate_pins = []
    checked_pin_count = 0
    for reference_pin in reference_pins:
        pin_id = reference_pin['pin_id']
        detected_pin = reference_to_detected_pin[pin_id]
        if detected_pin is None:
            continue
        checked_pin_count += 1
        reference_aspect_ratio = reference_pin['height'] / reference_pin['width']
        reference_fill_ratio = reference_pin['area'] / (reference_pin['width'] * reference_pin['height'])
        detected_aspect_ratio = detected_pin['height'] / detected_pin['width']
        detected_fill_ratio = detected_pin['area'] / (detected_pin['width'] * detected_pin['height'])
        aspect_ratio_error = abs(detected_aspect_ratio - reference_aspect_ratio) / reference_aspect_ratio
        fill_ratio_error = abs(detected_fill_ratio - reference_fill_ratio) / reference_fill_ratio
        is_aspect_abnormal = aspect_ratio_error > ASPECT_RATIO_ERROR_LIMIT
        is_fill_abnormal = fill_ratio_error > FILL_RATIO_ERROR_LIMIT
        is_candidate = is_aspect_abnormal or is_fill_abnormal
        if not is_candidate:
            continue
        candidate_pins.append({'pin_id': pin_id, 'reference_aspect_ratio': reference_aspect_ratio, 'detected_aspect_ratio': detected_aspect_ratio, 'aspect_ratio_error': aspect_ratio_error, 'is_aspect_abnormal': is_aspect_abnormal, 'reference_fill_ratio': reference_fill_ratio, 'detected_fill_ratio': detected_fill_ratio, 'fill_ratio_error': fill_ratio_error, 'is_fill_abnormal': is_fill_abnormal, 'x': detected_pin['x'], 'y': detected_pin['y'], 'width': detected_pin['width'], 'height': detected_pin['height']})
    return {'checked_pin_count': checked_pin_count, 'candidate_pins': candidate_pins}

def draw_match_result(result, reference_pins, match_result, body_box, color):
    body_x, body_y, body_w, body_h = body_box
    reference_pin_lookup = {}
    for reference_pin in reference_pins:
        reference_pin_lookup[reference_pin['pin_id']] = reference_pin
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    for pin_id, detected_pin in reference_to_detected_pin.items():
        if detected_pin is not None:
            x = detected_pin['x']
            y = detected_pin['y']
            bw = detected_pin['width']
            bh = detected_pin['height']
            cx = detected_pin['cx']
            cy = detected_pin['cy']
            cv2.rectangle(result, (x, y), (x + bw - 1, y + bh - 1), color, 1)
            cv2.circle(result, (int(round(cx)), int(round(cy))), 2, (0, 0, 255), -1)
            cv2.putText(result, pin_id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            reference_pin = reference_pin_lookup[pin_id]
            expected_x = body_x + reference_pin['relative_x'] * body_w
            expected_y = body_y + reference_pin['relative_y'] * body_h
            px = int(round(expected_x))
            py = int(round(expected_y))
            cv2.line(result, (px - 6, py - 6), (px + 6, py + 6), (0, 0, 255), 2)
            cv2.line(result, (px - 6, py + 6), (px + 6, py - 6), (0, 0, 255), 2)
            cv2.putText(result, pin_id + ' MISS', (px - 15, max(12, py - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

def draw_pitch_abnormalities(result, pitch_result):
    for pair in pitch_result['abnormal_pairs']:
        x1 = int(round(pair['left_cx']))
        y1 = int(round(pair['left_cy']))
        x2 = int(round(pair['right_cx']))
        y2 = int(round(pair['right_cy']))
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text_x = (x1 + x2) // 2
        text_y = min(y1, y2) - 8
        cv2.putText(result, 'PITCH', (text_x - 12, max(12, text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

def draw_position_shifts(result, position_result):
    for pin in position_result['shifted_pins']:
        x = pin['x']
        y = pin['y']
        bw = pin['width']
        bh = pin['height']
        cv2.rectangle(result, (x - 2, y - 2), (x + bw + 1, y + bh + 1), (0, 0, 255), 2)
        cv2.putText(result, pin['pin_id'] + ' SHIFT', (x - 10, max(12, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

def draw_size_abnormalities(result, size_result):
    for pin in size_result['abnormal_pins']:
        x = pin['x']
        y = pin['y']
        bw = pin['width']
        bh = pin['height']
        cv2.rectangle(result, (x - 3, y - 3), (x + bw + 2, y + bh + 2), (0, 0, 255), 2)
        cv2.putText(result, pin['pin_id'] + ' SIZE', (x - 10, max(12, y - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

def draw_shape_candidates(result, shape_result):
    for pin in shape_result['candidate_pins']:
        x = pin['x']
        y = pin['y']
        bw = pin['width']
        bh = pin['height']
        cv2.rectangle(result, (x - 4, y - 4), (x + bw + 3, y + bh + 3), (255, 0, 255), 2)
        cv2.putText(result, pin['pin_id'] + ' SHAPE', (x - 10, max(12, y - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

def crop_fixed_pin_patch(img, cx, cy, patch_width, patch_height):
    img_h, img_w = img.shape[:2]
    center_x = int(round(cx))
    center_y = int(round(cy))
    x1 = center_x - patch_width // 2
    y1 = center_y - patch_height // 2
    x2 = x1 + patch_width
    y2 = y1 + patch_height
    patch = np.zeros((patch_height, patch_width, 3), dtype=img.dtype)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(img_w, x2)
    src_y2 = min(img_h, y2)
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    return patch

def preprocess_pin_patch_for_onnx(patch_bgr):
    patch_h, patch_w = patch_bgr.shape[:2]
    if patch_h > patch_w:
        difference = patch_h - patch_w
        left = difference // 2
        right = difference - left
        square_bgr = cv2.copyMakeBorder(patch_bgr, 0, 0, left, right, cv2.BORDER_REPLICATE)
    elif patch_w > patch_h:
        difference = patch_w - patch_h
        top = difference // 2
        bottom = difference - top
        square_bgr = cv2.copyMakeBorder(patch_bgr, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
    else:
        square_bgr = patch_bgr
    resized_bgr = cv2.resize(square_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    image_array = resized_rgb.astype(np.float32) / 255.0
    image_array = (image_array - CNN_MEAN) / CNN_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)

def softmax_numpy(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def create_onnx_session():
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f'ONNX model not found: {ONNX_MODEL_PATH}')
    return ort.InferenceSession(str(ONNX_MODEL_PATH), providers=[RECIPE['cnn']['provider']])

def run_cnn_diagnostic_for_row(corrected_img, reference_pins, match_result, test_body_box, session):
    reference_to_detected_pin = match_result['reference_to_detected_pin']
    input_name = session.get_inputs()[0].name
    results = []
    body_x, body_y, body_w, body_h = test_body_box
    for reference_pin in reference_pins:
        pin_id = reference_pin['pin_id']
        detected_pin = reference_to_detected_pin[pin_id]
        if detected_pin is None:
            results.append({'pin_id': pin_id, 'status': 'missing_skipped', 'p_normal': None, 'p_defect': None, 'prediction': None, 'x': None, 'y': None, 'width': None, 'height': None})
            continue
        slot_cx = body_x + reference_pin['relative_x'] * body_w
        slot_cy = body_y + reference_pin['relative_y'] * body_h
        patch = crop_fixed_pin_patch(corrected_img, slot_cx, slot_cy, CNN_PATCH_WIDTH, CNN_PATCH_HEIGHT)
        input_array = preprocess_pin_patch_for_onnx(patch)
        logits = session.run(None, {input_name: input_array})[0]
        probabilities = softmax_numpy(logits)
        p_normal = float(probabilities[0, 0])
        p_defect = float(probabilities[0, 1])
        prediction = 'defect' if p_defect >= CNN_DEFECT_THRESHOLD else 'normal'
        results.append({'pin_id': pin_id, 'status': 'checked', 'p_normal': p_normal, 'p_defect': p_defect, 'prediction': prediction, 'x': detected_pin['x'], 'y': detected_pin['y'], 'width': detected_pin['width'], 'height': detected_pin['height']})
    return results

def draw_cnn_diagnostic(img, cnn_results):
    result = img.copy()
    for item in cnn_results:
        if item['status'] != 'checked':
            continue
        x = int(item['x'])
        y = int(item['y'])
        width = int(item['width'])
        height = int(item['height'])
        if item['prediction'] == 'defect':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), color, 1)
    return result

def main():
    reference_img = read_image(REFERENCE_IMAGE)
    test_img = read_image(TEST_IMAGE)
    print('=' * 72)
    print('IC PIN HYBRID INSPECTOR')
    print('=' * 72)
    print()
    print(f"Recipe product = {RECIPE['product_name']}")
    print(f"Recipe version = {RECIPE['recipe_version']}")
    print(f'Body threshold = {BODY_THRESHOLD}')
    print(f'Expected pins/row = {EXPECTED_PIN_COUNT}')
    print()
    print('Reference image:')
    print(REFERENCE_IMAGE)
    print()
    print('Original test image:')
    print(TEST_IMAGE)
    test_deskew_result = deskew_image(test_img)
    corrected_test_img = test_deskew_result['image']
    print()
    print('Deskew Information:')
    print(f"Raw angle = {test_deskew_result['raw_angle']:.3f} deg")
    print(f"Normalized angle = {test_deskew_result['normalized_angle']:.3f} deg")
    save_image(corrected_test_img, DESKEWED_TEST_IMAGE)
    print()
    print('Deskewed test image saved to:')
    print(DESKEWED_TEST_IMAGE)
    reference_image_data = process_image(reference_img)
    test_image_data = process_image(corrected_test_img)
    print()
    print('Reference Pin Count:')
    print('Top =', len(reference_image_data['top_raw_pins']))
    print('Bottom =', len(reference_image_data['bottom_raw_pins']))
    if len(reference_image_data['top_raw_pins']) != EXPECTED_PIN_COUNT:
        raise RuntimeError(f'Reference Top Pin count is not {EXPECTED_PIN_COUNT}.')
    if len(reference_image_data['bottom_raw_pins']) != EXPECTED_PIN_COUNT:
        raise RuntimeError(f'Reference Bottom Pin count is not {EXPECTED_PIN_COUNT}.')
    top_reference_pins = build_pin_records(reference_image_data['top_raw_pins'], reference_image_data['body_box'], 'T')
    bottom_reference_pins = build_pin_records(reference_image_data['bottom_raw_pins'], reference_image_data['body_box'], 'B')
    top_detected_pins = build_pin_records(test_image_data['top_raw_pins'], test_image_data['body_box'])
    bottom_detected_pins = build_pin_records(test_image_data['bottom_raw_pins'], test_image_data['body_box'])
    top_match_result = match_detected_pins_to_reference(top_reference_pins, top_detected_pins)
    bottom_match_result = match_detected_pins_to_reference(bottom_reference_pins, bottom_detected_pins)
    top_missing_gap_result = check_missing_gap_evidence(top_reference_pins, top_match_result, test_image_data['body_box'])
    bottom_missing_gap_result = check_missing_gap_evidence(bottom_reference_pins, bottom_match_result, test_image_data['body_box'])
    top_pitch_result = check_pitch_abnormalities(top_reference_pins, top_match_result, test_image_data['body_box'])
    bottom_pitch_result = check_pitch_abnormalities(bottom_reference_pins, bottom_match_result, test_image_data['body_box'])
    top_position_result = check_position_shifts(top_reference_pins, top_match_result, test_image_data['body_box'])
    bottom_position_result = check_position_shifts(bottom_reference_pins, bottom_match_result, test_image_data['body_box'])
    top_size_result = check_pin_size_abnormalities(top_reference_pins, top_match_result)
    bottom_size_result = check_pin_size_abnormalities(bottom_reference_pins, bottom_match_result)
    top_shape_result = check_shape_candidates(top_reference_pins, top_match_result)
    bottom_shape_result = check_shape_candidates(bottom_reference_pins, bottom_match_result)
    cnn_session = create_onnx_session()
    top_cnn_results = run_cnn_diagnostic_for_row(corrected_test_img, top_reference_pins, top_match_result, test_image_data['body_box'], cnn_session)
    bottom_cnn_results = run_cnn_diagnostic_for_row(corrected_test_img, bottom_reference_pins, bottom_match_result, test_image_data['body_box'], cnn_session)
    all_cnn_results = top_cnn_results + bottom_cnn_results
    cnn_checked_count = sum((1 for item in all_cnn_results if item['status'] == 'checked'))
    cnn_missing_skipped_count = sum((1 for item in all_cnn_results if item['status'] == 'missing_skipped'))
    cnn_defect_candidates = [item for item in all_cnn_results if item['status'] == 'checked' and item['prediction'] == 'defect']
    cnn_diagnostic_image = draw_cnn_diagnostic(corrected_test_img, all_cnn_results)
    save_image(cnn_diagnostic_image, CNN_DIAGNOSTIC_IMAGE)
    top_missing_pin_ids = top_match_result['missing_pin_ids']
    bottom_missing_pin_ids = bottom_match_result['missing_pin_ids']
    top_unmatched_pins = top_match_result['unmatched_pins']
    bottom_unmatched_pins = bottom_match_result['unmatched_pins']
    print()
    print('Test Pin Count:')
    print('Top =', len(top_detected_pins))
    print('Bottom =', len(bottom_detected_pins))
    print()
    print('Missing Pin Result:')
    if top_missing_pin_ids:
        print('Top missing:', ', '.join(top_missing_pin_ids))
    else:
        print('Top missing: None')
    if bottom_missing_pin_ids:
        print('Bottom missing:', ', '.join(bottom_missing_pin_ids))
    else:
        print('Bottom missing: None')
    print()
    print('Matching Information:')
    print(f"Top reference pitch ratio = {top_match_result['reference_pitch_ratio']:.4f}")
    print(f"Top match tolerance ratio = {top_match_result['match_tolerance_ratio']:.4f}")
    print(f"Bottom reference pitch ratio = {bottom_match_result['reference_pitch_ratio']:.4f}")
    print(f"Bottom match tolerance ratio = {bottom_match_result['match_tolerance_ratio']:.4f}")
    print()
    print('Unmatched Detected Pins:')
    print('Top unmatched =', len(top_unmatched_pins))
    print('Bottom unmatched =', len(bottom_unmatched_pins))
    print()
    print('Missing Gap Evidence:')
    if top_missing_gap_result['evidence']:
        for evidence in top_missing_gap_result['evidence']:
            missing_text = ', '.join(evidence['missing_pin_ids'])
            if evidence['has_two_sided_evidence']:
                print(f"Top {missing_text}: {evidence['left_pin_id']}-{evidence['right_pin_id']}, gap={evidence['detected_gap_px']:.2f} px, expected={evidence['expected_gap_px']:.2f} px, error={evidence['error_ratio'] * 100:.1f}%, consistent={evidence['is_consistent']}")
            else:
                print(f'Top {missing_text}: no two-sided gap evidence')
    else:
        print('Top missing gap evidence: None')
    if bottom_missing_gap_result['evidence']:
        for evidence in bottom_missing_gap_result['evidence']:
            missing_text = ', '.join(evidence['missing_pin_ids'])
            if evidence['has_two_sided_evidence']:
                print(f"Bottom {missing_text}: {evidence['left_pin_id']}-{evidence['right_pin_id']}, gap={evidence['detected_gap_px']:.2f} px, expected={evidence['expected_gap_px']:.2f} px, error={evidence['error_ratio'] * 100:.1f}%, consistent={evidence['is_consistent']}")
            else:
                print(f'Bottom {missing_text}: no two-sided gap evidence')
    else:
        print('Bottom missing gap evidence: None')
    print()
    print('Pitch Abnormal Result:')
    print('Top checked adjacent pairs =', top_pitch_result['checked_pair_count'])
    if top_pitch_result['abnormal_pairs']:
        for pair in top_pitch_result['abnormal_pairs']:
            print(f"Top abnormal: {pair['left_pin_id']}-{pair['right_pin_id']}, pitch={pair['detected_pitch_px']:.2f} px, expected={pair['expected_pitch_px']:.2f} px, error={pair['error_ratio'] * 100:.1f}%")
    else:
        print('Top abnormal: None')
    print('Bottom checked adjacent pairs =', bottom_pitch_result['checked_pair_count'])
    if bottom_pitch_result['abnormal_pairs']:
        for pair in bottom_pitch_result['abnormal_pairs']:
            print(f"Bottom abnormal: {pair['left_pin_id']}-{pair['right_pin_id']}, pitch={pair['detected_pitch_px']:.2f} px, expected={pair['expected_pitch_px']:.2f} px, error={pair['error_ratio'] * 100:.1f}%")
    else:
        print('Bottom abnormal: None')
    print()
    print('Position Shift Result:')
    print(f"Top reference pitch = {top_position_result['reference_pitch_px']:.2f} px")
    print(f"Top detected pitch = {top_position_result['detected_pitch_px']:.2f} px")
    print(f"Top alignment scale = {top_position_result['scale_x']:.4f}")
    print(f"Top shift limit = {top_position_result['shift_limit_px']:.2f} px")
    if top_position_result['shifted_pins']:
        for pin in top_position_result['shifted_pins']:
            print(f"Top shifted: {pin['pin_id']}, dx={pin['shift_x_px']:.2f} px, dy={pin['shift_y_px']:.2f} px")
    else:
        print('Top shifted: None')
    print(f"Bottom reference pitch = {bottom_position_result['reference_pitch_px']:.2f} px")
    print(f"Bottom detected pitch = {bottom_position_result['detected_pitch_px']:.2f} px")
    print(f"Bottom alignment scale = {bottom_position_result['scale_x']:.4f}")
    print(f"Bottom shift limit = {bottom_position_result['shift_limit_px']:.2f} px")
    if bottom_position_result['shifted_pins']:
        for pin in bottom_position_result['shifted_pins']:
            print(f"Bottom shifted: {pin['pin_id']}, dx={pin['shift_x_px']:.2f} px, dy={pin['shift_y_px']:.2f} px")
    else:
        print('Bottom shifted: None')
    print()
    print('Pin Size Abnormal Result:')
    print('Top checked pins =', top_size_result['checked_pin_count'])
    if top_size_result['abnormal_pins']:
        for pin in top_size_result['abnormal_pins']:
            print(f"Top abnormal: {pin['pin_id']}, width={pin['detected_width']} px (ref={pin['reference_width']} px, error={pin['width_error_ratio'] * 100:.1f}%), height={pin['detected_height']} px (ref={pin['reference_height']} px, error={pin['height_error_ratio'] * 100:.1f}%), area={pin['detected_area']} (ref={pin['reference_area']}, error={pin['area_error_ratio'] * 100:.1f}%)")
    else:
        print('Top size abnormal: None')
    print('Bottom checked pins =', bottom_size_result['checked_pin_count'])
    if bottom_size_result['abnormal_pins']:
        for pin in bottom_size_result['abnormal_pins']:
            print(f"Bottom abnormal: {pin['pin_id']}, width={pin['detected_width']} px (ref={pin['reference_width']} px, error={pin['width_error_ratio'] * 100:.1f}%), height={pin['detected_height']} px (ref={pin['reference_height']} px, error={pin['height_error_ratio'] * 100:.1f}%), area={pin['detected_area']} (ref={pin['reference_area']}, error={pin['area_error_ratio'] * 100:.1f}%)")
    else:
        print('Bottom size abnormal: None')
    print()
    print('Shape Candidate Result:')
    print('Top checked pins =', top_shape_result['checked_pin_count'])
    if top_shape_result['candidate_pins']:
        for pin in top_shape_result['candidate_pins']:
            print(f"Top candidate: {pin['pin_id']}, aspect error={pin['aspect_ratio_error'] * 100:.1f}%, fill error={pin['fill_ratio_error'] * 100:.1f}%")
    else:
        print('Top shape candidate: None')
    print('Bottom checked pins =', bottom_shape_result['checked_pin_count'])
    if bottom_shape_result['candidate_pins']:
        for pin in bottom_shape_result['candidate_pins']:
            print(f"Bottom candidate: {pin['pin_id']}, aspect error={pin['aspect_ratio_error'] * 100:.1f}%, fill error={pin['fill_ratio_error'] * 100:.1f}%")
    else:
        print('Bottom shape candidate: None')
    print()
    print('=' * 72)
    print('CNN APPEARANCE BRANCH')
    print('=' * 72)
    print()
    print(f'ONNX model = {ONNX_MODEL_PATH}')
    print(f'Provider = {cnn_session.get_providers()}')
    print(f'CNN patch = {CNN_PATCH_WIDTH} x {CNN_PATCH_HEIGHT}')
    print('CNN crop mode = Reference-slot aligned')
    print(f'CNN diagnostic threshold = {CNN_DEFECT_THRESHOLD:.2f}')
    print()
    print('Per-Pin CNN Results:')
    for item in all_cnn_results:
        if item['status'] == 'missing_skipped':
            print(f"{item['pin_id']}: Missing -> CNN skipped")
            continue
        print(f"{item['pin_id']}: P(normal)={item['p_normal']:.4f}, P(defect)={item['p_defect']:.4f}, CNN={item['prediction']}")
    print()
    print(f'CNN checked pins = {cnn_checked_count}')
    print(f'CNN missing-skipped pins = {cnn_missing_skipped_count}')
    if cnn_defect_candidates:
        print('CNN defect candidates = ' + ', '.join((item['pin_id'] for item in cnn_defect_candidates)))
    else:
        print('CNN defect candidates = None')
    print()
    print('CNN diagnostic image saved to:')
    print(CNN_DIAGNOSTIC_IMAGE)
    print()
    print('CNN branch included in final fusion.')
    rule_is_ng = bool(top_missing_pin_ids or bottom_missing_pin_ids or top_unmatched_pins or bottom_unmatched_pins or top_pitch_result['abnormal_pairs'] or bottom_pitch_result['abnormal_pairs'] or top_position_result['shifted_pins'] or bottom_position_result['shifted_pins'] or top_size_result['abnormal_pins'] or bottom_size_result['abnormal_pins'] or top_shape_result['candidate_pins'] or bottom_shape_result['candidate_pins'])
    cnn_is_ng = bool(cnn_defect_candidates)
    is_ng = rule_is_ng or cnn_is_ng
    print()
    print('=' * 72)
    print('HYBRID FUSION')
    print('=' * 72)
    print('Rule Result = ' + ('NG' if rule_is_ng else 'OK'))
    print('CNN Result  = ' + ('NG' if cnn_is_ng else 'OK'))
    if cnn_defect_candidates:
        print('CNN NG Pins = ' + ', '.join((item['pin_id'] for item in cnn_defect_candidates)))
    else:
        print('CNN NG Pins = None')
    print('Fusion Rule = Rule NG OR CNN NG')
    print('Final Result = ' + ('NG' if is_ng else 'OK'))
    result = corrected_test_img.copy()
    body_x, body_y, body_w, body_h = test_image_data['body_box']
    top_x1, top_y1, top_x2, top_y2 = test_image_data['top_box']
    bottom_x1, bottom_y1, bottom_x2, bottom_y2 = test_image_data['bottom_box']
    cv2.rectangle(result, (body_x, body_y), (body_x + body_w - 1, body_y + body_h - 1), (255, 0, 0), 2)
    cv2.rectangle(result, (top_x1, top_y1), (top_x2 - 1, top_y2 - 1), (0, 255, 0), 1)
    cv2.rectangle(result, (bottom_x1, bottom_y1), (bottom_x2 - 1, bottom_y2 - 1), (0, 255, 255), 1)
    draw_match_result(result, top_reference_pins, top_match_result, test_image_data['body_box'], (0, 255, 0))
    draw_match_result(result, bottom_reference_pins, bottom_match_result, test_image_data['body_box'], (0, 255, 255))
    draw_pitch_abnormalities(result, top_pitch_result)
    draw_pitch_abnormalities(result, bottom_pitch_result)
    draw_position_shifts(result, top_position_result)
    draw_position_shifts(result, bottom_position_result)
    draw_size_abnormalities(result, top_size_result)
    draw_size_abnormalities(result, bottom_size_result)
    draw_shape_candidates(result, top_shape_result)
    draw_shape_candidates(result, bottom_shape_result)
    for item in cnn_defect_candidates:
        x = int(item['x'])
        y = int(item['y'])
        width = int(item['width'])
        height = int(item['height'])
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (255, 0, 255), 2)
        cv2.putText(result, 'CNN', (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
    status_text = 'NG' if is_ng else 'OK'
    status_color = (0, 0, 255) if is_ng else (0, 255, 0)
    cv2.putText(result, 'Result: ' + status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    save_image(result, OUTPUT_IMAGE)
    print()
    print('Result image saved to:')
    print(OUTPUT_IMAGE)
if __name__ == '__main__':
    main()
