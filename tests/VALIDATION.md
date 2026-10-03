# Validation

## Core 8-case regression

Run:

```powershell
python -m pip install -r requirements.txt
python .\src\batch_test.py
```

Expected summary:

```text
Regression = 8/8 PASS
Overall = PASS
```

The cleaned GitHub version was run in the target Windows environment with ONNX Runtime 1.30.0 and returned the expected result for all eight cases.

The regression checks:

- normal sample -> OK
- missing top pin -> NG
- missing bottom pin -> NG
- position shift -> NG
- width abnormality -> NG
- height abnormality -> NG
- area abnormality -> NG
- shape abnormality -> NG

## Real anomaly-detection extension

The PatchCore-style full MVTec AD `transistor` evaluation was run with 213 `train/good` images and the complete 100-image test split.

Key results:

```text
Image-level AUROC = 0.937083
Image-level AP    = 0.954614
Pixel-level AUROC = 0.841665
Pixel-level AP    = 0.478352
```

See `docs/MVTEC_TRANSISTOR_RESULTS.md` for category-level results and limitations.
