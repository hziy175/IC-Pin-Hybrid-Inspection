from pathlib import Path
import argparse
import csv
import re
import shutil
import subprocess
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
INPUT_DIR = ROOT_DIR / 'data' / 'input'
OUTPUT_DIR = ROOT_DIR / 'output'
BATCH_DIR = OUTPUT_DIR / 'batch'
INSPECTOR_PATH = SRC_DIR / 'main.py'
CSV_PATH = BATCH_DIR / 'inspection_results.csv'
LOG_DIR = BATCH_DIR / 'logs'
RESULT_IMAGE_DIR = BATCH_DIR / 'result_images'
REGRESSION_CASES = [('test_chip.jpg', 'OK'), ('defect_missing_top.jpg', 'NG'), ('defect_missing_bottom.jpg', 'NG'), ('defect_shift_top_T08_right5.jpg', 'NG'), ('defect_width_top_T08.jpg', 'NG'), ('defect_height_top_T08.jpg', 'NG'), ('defect_area_top_T08.jpg', 'NG'), ('defect_bent_top_T08.jpg', 'NG')]

def find_one(pattern, text, default=''):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return default
    return match.group(1).strip()

def get_section(text, start_title, end_title=None):
    start_index = text.find(start_title)
    if start_index < 0:
        return ''
    if end_title is None:
        return text[start_index:]
    end_index = text.find(end_title, start_index + len(start_title))
    if end_index < 0:
        return text[start_index:]
    return text[start_index:end_index]

def parse_result(stdout_text):
    rule_result = find_one('^Rule Result\\s*=\\s*(OK|NG)\\s*$', stdout_text)
    cnn_result = find_one('^CNN Result\\s*=\\s*(OK|NG)\\s*$', stdout_text)
    cnn_ng_pins = find_one('^CNN NG Pins\\s*=\\s*(.+?)\\s*$', stdout_text, default='None')
    final_result = find_one('^Final Result\\s*=\\s*(OK|NG)\\s*$', stdout_text)
    missing_section = get_section(stdout_text, 'Missing Pin Result:', 'Matching Information:')
    missing_top = find_one('^Top missing:\\s*(.+?)\\s*$', missing_section, default='None')
    missing_bottom = find_one('^Bottom missing:\\s*(.+?)\\s*$', missing_section, default='None')
    pitch_section = get_section(stdout_text, 'Pitch Abnormal Result:', 'Position Shift Result:')
    pitch_top = find_one('^Top abnormal:\\s*(.+?)\\s*$', pitch_section, default='None')
    pitch_bottom = find_one('^Bottom abnormal:\\s*(.+?)\\s*$', pitch_section, default='None')
    position_section = get_section(stdout_text, 'Position Shift Result:', 'Pin Size Abnormal Result:')
    position_top = find_one('^Top shifted:\\s*(.+?)\\s*$', position_section, default='None')
    position_bottom = find_one('^Bottom shifted:\\s*(.+?)\\s*$', position_section, default='None')
    size_section = get_section(stdout_text, 'Pin Size Abnormal Result:', 'Shape Candidate Result:')
    size_top = find_one('^(?:Top abnormal|Top size abnormal):\\s*(.+?)\\s*$', size_section, default='None')
    size_bottom = find_one('^(?:Bottom abnormal|Bottom size abnormal):\\s*(.+?)\\s*$', size_section, default='None')
    shape_section = get_section(stdout_text, 'Shape Candidate Result:', 'CNN APPEARANCE BRANCH')
    shape_top = find_one('^(?:Top candidate|Top shape candidate):\\s*(.+?)\\s*$', shape_section, default='None')
    shape_bottom = find_one('^(?:Bottom candidate|Bottom shape candidate):\\s*(.+?)\\s*$', shape_section, default='None')
    return {'rule_result': rule_result, 'cnn_result': cnn_result, 'cnn_ng_pins': cnn_ng_pins, 'final_result': final_result, 'missing_top': missing_top, 'missing_bottom': missing_bottom, 'pitch_top': pitch_top, 'pitch_bottom': pitch_bottom, 'position_top': position_top, 'position_bottom': position_bottom, 'size_top': size_top, 'size_bottom': size_bottom, 'shape_top': shape_top, 'shape_bottom': shape_bottom}

def run_one_case(image_name, expected_result=''):
    image_path = INPUT_DIR / image_name
    if not image_path.exists():
        return {'image': image_name, 'expected': expected_result, 'rule_result': '', 'cnn_result': '', 'cnn_ng_pins': '', 'final_result': '', 'regression_pass': 'FAIL', 'error': f'Input image not found: {image_path}'}
    command = [sys.executable, str(INSPECTOR_PATH), image_name]
    completed = subprocess.run(command, cwd=SRC_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace')
    full_log = completed.stdout
    if completed.stderr:
        full_log += '\n\n[STDERR]\n'
        full_log += completed.stderr
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'{Path(image_name).stem}.txt'
    log_path.write_text(full_log, encoding='utf-8')
    parsed = parse_result(completed.stdout)
    source_result_image = OUTPUT_DIR / 'results' / f'{Path(image_name).stem}_result.jpg'
    RESULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    copied_result_image = RESULT_IMAGE_DIR / f'{Path(image_name).stem}_result.jpg'
    if source_result_image.exists():
        shutil.copy2(source_result_image, copied_result_image)
        result_image_text = str(copied_result_image)
    else:
        result_image_text = ''
    if completed.returncode != 0:
        regression_pass = 'FAIL'
        error_text = f'Inspector exit code = {completed.returncode}'
    elif not parsed['final_result']:
        regression_pass = 'FAIL'
        error_text = 'Could not parse Final Result'
    else:
        if expected_result:
            regression_pass = 'PASS' if parsed['final_result'] == expected_result else 'FAIL'
        else:
            regression_pass = ''
        error_text = ''
    result = {'image': image_name, 'expected': expected_result, **parsed, 'regression_pass': regression_pass, 'result_image': result_image_text, 'log_file': str(log_path), 'error': error_text}
    return result

def write_csv(rows):
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ['image', 'expected', 'rule_result', 'cnn_result', 'cnn_ng_pins', 'final_result', 'regression_pass', 'missing_top', 'missing_bottom', 'pitch_top', 'pitch_bottom', 'position_top', 'position_bottom', 'size_top', 'size_bottom', 'shape_top', 'shape_bottom', 'result_image', 'log_file', 'error']
    with CSV_PATH.open('w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

def print_summary(rows):
    print()
    print('=' * 100)
    print('BATCH INSPECTION SUMMARY')
    print('=' * 100)
    print()
    print(f"{'Image':<36} {'Expected':<9} {'Rule':<7} {'CNN':<7} {'Final':<7} {'Check':<7}")
    print('-' * 100)
    for row in rows:
        print(f"{row['image']:<36} {row.get('expected', ''):<9} {row.get('rule_result', ''):<7} {row.get('cnn_result', ''):<7} {row.get('final_result', ''):<7} {row.get('regression_pass', ''):<7}")
    regression_rows = [row for row in rows if row.get('expected')]
    if regression_rows:
        passed = sum((1 for row in regression_rows if row.get('regression_pass') == 'PASS'))
        total = len(regression_rows)
        print()
        print(f'Regression = {passed}/{total} PASS')
        if passed == total:
            print('Overall = PASS')
        else:
            print('Overall = FAIL')
    print()
    print(f'CSV saved to:')
    print(CSV_PATH)
    print()
    print(f'Per-image logs:')
    print(LOG_DIR)
    print()
    print(f'Collected result images:')
    print(RESULT_IMAGE_DIR)

def collect_all_jpg_cases():
    image_paths = sorted(INPUT_DIR.glob('*.jpg'))
    return [(image_path.name, '') for image_path in image_paths]

def main():
    parser = argparse.ArgumentParser(description='Batch runner for the hybrid IC pin inspector.')
    parser.add_argument('--all', action='store_true', help='Inspect all JPG files in data/input. Without --all, run the validated 8-case regression.')
    args = parser.parse_args()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        cases = collect_all_jpg_cases()
        mode_name = 'ALL JPG FILES'
    else:
        cases = REGRESSION_CASES
        mode_name = '8-CASE REGRESSION'
    print('=' * 100)
    print('BATCH INSPECTION')
    print('=' * 100)
    print()
    print(f'Mode = {mode_name}')
    print(f'Inspector = {INSPECTOR_PATH}')
    print(f'Case count = {len(cases)}')
    rows = []
    for index, (image_name, expected_result) in enumerate(cases, start=1):
        print()
        print(f'[{index}/{len(cases)}] {image_name}')
        row = run_one_case(image_name, expected_result)
        rows.append(row)
        print(f"Rule={row.get('rule_result', '')}, CNN={row.get('cnn_result', '')}, Final={row.get('final_result', '')}, Check={row.get('regression_pass', '')}")
        if row.get('error'):
            print(f"ERROR: {row['error']}")
    write_csv(rows)
    print_summary(rows)
if __name__ == '__main__':
    main()
