from pathlib import Path
import json
ROOT_DIR = Path(__file__).resolve().parents[1]
RECIPE_PATH = Path(__file__).resolve().parent / 'recipe_ic_a.json'

def load_recipe(recipe_path=RECIPE_PATH):
    with open(recipe_path, 'r', encoding='utf-8') as f:
        recipe = json.load(f)
    validate_recipe(recipe)
    return recipe

def validate_recipe(recipe):
    body = recipe['body_detection']
    pin_roi = recipe['pin_roi']
    pin_candidate = recipe['pin_candidate']
    pin_layout = recipe['pin_layout']
    matching = recipe['matching']
    defect = recipe['defect_limits']
    assert 0 <= body['threshold'] <= 255
    assert 0 < body['min_area_ratio'] < body['max_area_ratio'] <= 1
    assert 0 <= body['center_min_ratio'] < body['center_max_ratio'] <= 1
    assert 0 < body['min_rectangularity'] <= 1
    assert 0 <= pin_roi['pin_threshold'] <= 255
    assert 0 < pin_roi['pin_band_ratio'] < 1
    assert 0 <= pin_roi['x_margin_ratio'] < 1
    assert 0 <= pin_roi['y_margin_ratio'] < 1
    assert 0 < pin_candidate['min_area'] < pin_candidate['max_area']
    assert 0 < pin_candidate['min_aspect'] < pin_candidate['max_aspect']
    assert isinstance(pin_layout['expected_pin_count_per_row'], int)
    assert pin_layout['expected_pin_count_per_row'] > 0
    assert 0 < matching['tolerance_pitch_ratio'] < 1
    for key, value in defect.items():
        assert value > 0, f'{key} must be > 0'
    assert 'geometric_reference_image' in recipe['reference']
    return True
RECIPE = load_recipe()
BODY_THRESHOLD = RECIPE['body_detection']['threshold']
MIN_BODY_AREA_RATIO = RECIPE['body_detection']['min_area_ratio']
MAX_BODY_AREA_RATIO = RECIPE['body_detection']['max_area_ratio']
BODY_CENTER_MIN_RATIO = RECIPE['body_detection']['center_min_ratio']
BODY_CENTER_MAX_RATIO = RECIPE['body_detection']['center_max_ratio']
MIN_BODY_RECTANGULARITY = RECIPE['body_detection']['min_rectangularity']
PIN_THRESHOLD = RECIPE['pin_roi']['pin_threshold']
PIN_BAND_RATIO = RECIPE['pin_roi']['pin_band_ratio']
X_MARGIN_RATIO = RECIPE['pin_roi']['x_margin_ratio']
Y_MARGIN_RATIO = RECIPE['pin_roi']['y_margin_ratio']
MIN_PIN_AREA = RECIPE['pin_candidate']['min_area']
MAX_PIN_AREA = RECIPE['pin_candidate']['max_area']
MIN_PIN_ASPECT = RECIPE['pin_candidate']['min_aspect']
MAX_PIN_ASPECT = RECIPE['pin_candidate']['max_aspect']
EXPECTED_PIN_COUNT = RECIPE['pin_layout']['expected_pin_count_per_row']
MATCH_TOLERANCE_RATIO = RECIPE['matching']['tolerance_pitch_ratio']
MISSING_GAP_ERROR_LIMIT = RECIPE['defect_limits']['missing_gap_error_limit']
PITCH_ERROR_LIMIT = RECIPE['defect_limits']['pitch_error_limit']
POSITION_SHIFT_LIMIT_PITCH_RATIO = RECIPE['defect_limits']['position_shift_pitch_ratio']
WIDTH_ERROR_LIMIT = RECIPE['defect_limits']['width_error_limit']
HEIGHT_ERROR_LIMIT = RECIPE['defect_limits']['height_error_limit']
AREA_ERROR_LIMIT = RECIPE['defect_limits']['area_error_limit']
ASPECT_RATIO_ERROR_LIMIT = RECIPE['defect_limits']['aspect_ratio_error_limit']
FILL_RATIO_ERROR_LIMIT = RECIPE['defect_limits']['fill_ratio_error_limit']
GEOMETRIC_REFERENCE_IMAGE = RECIPE['reference']['geometric_reference_image']
