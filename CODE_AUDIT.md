# Code Audit and Cleanup Notes

This package was produced from the uploaded `practice.zip` after reviewing the full source tree and extracting the validated project path into a GitHub-oriented non-learning version.

## Audit scope

- 76 Python files in the original archive were syntax-compiled successfully.
- The archive contained multiple Day1-Day12 teaching, diagnostic, debug, intermediate, and duplicated variants.
- The validated release path was identified as the Day10 hybrid inspector + Day11 8-case regression + Day12 release package.
- Supplemental PatchCore, YOLO, and Open3D quick experiments were also reviewed and retained as separate experiments.

## Problems found in the original archive

1. **Large amount of duplicated learning/debug code**
   - Multiple versions of Day5-Day10 scripts coexisted.
   - Several files were intentionally diagnostic or teaching-only and were not appropriate as GitHub entry points.

2. **Release requirements were incomplete**
   - The old `utils.py` imported `pandas`, but the release `requirements.txt` did not include pandas.
   - The final inspector only needed `read_image` / `save_image`, so the cleaned version removes the pandas dependency instead of adding an unnecessary package.

3. **Old config contained unrelated and misleading fields**
   - The old `config.py` still contained legacy contour parameters and `PIXEL_SCALE_MM_PER_PX = 0.05`.
   - No real camera calibration was performed, so exposing a millimeter scale in the final GitHub version would be misleading.
   - The cleaned config only contains active paths and model locations.

4. **Learning-stage naming leaked into release code**
   - Runtime output contained labels such as `DAY7-7B` and `DAY10-2`.
   - Final files are renamed around their functional role rather than learning day numbers.

5. **Verbose matching debug was always executed**
   - The previous final inspector called a dedicated matching debug printer on every run.
   - The cleaned GitHub version removes that debug-only call while keeping the matching algorithm unchanged.

6. **Reference/model paths were stored under `output/`**
   - The release used `output/day3_corrected.jpg` as a reference and `output/day9/...onnx` as the model path.
   - The cleaned layout moves immutable assets to `data/reference/` and `models/`.

7. **Shape candidate naming could overstate physical semantics**
   - The old wording used `Bent / Damaged Candidate` although the implemented logic is based on aspect-ratio and fill-ratio deviations.
   - The final code uses the more accurate name `Shape Candidate`.

8. **Brightness robustness claim required qualification**
   - Brightness 0.8x / 1.2x passed only after a separate reference-guided global brightness normalization step.
   - That normalization is not integrated into the final main inspector, so the README now states this explicitly.

9. **Absolute Windows paths existed in old experimental scripts**
   - These were excluded from the final package.
   - The final package uses `Path(__file__)`-relative paths only.

## Preserved behavior

The core inspection thresholds and validated geometry/CNN fusion behavior were not changed:

- Body threshold: 95
- Pin threshold: 150
- Expected pins per row: 16
- Matching tolerance: 0.45 pitch
- Fusion: Rule NG OR CNN NG
- CNN threshold: 0.50

## Regression equivalence check

The cleaned package was compared against the previously validated release on all 8 regression images. Rule result, CNN result, CNN NG pins, and final result were identical for all 8 cases.

The cleaned batch runner also returned:

```text
Regression = 8/8 PASS
Overall = PASS
```

Because the execution environment used for this cleanup did not contain the `onnxruntime` Python package and had no network access, this equivalence run used an OpenCV-DNN compatibility adapter to execute the same ONNX model. The user's own Windows environment had already validated the original release with ONNX Runtime. The final ZIP should therefore be run once more on the user's local environment with the real `onnxruntime` dependency before public GitHub upload.
