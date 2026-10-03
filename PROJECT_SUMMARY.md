# Project Summary

## Main system

The main project is an offline IC pin inspection prototype with two complementary branches.

### Geometry branch

1. Detect the IC body by thresholding and contour filtering.
2. Estimate body orientation with `minAreaRect` and deskew the image.
3. Generate top and bottom pin ROIs from body geometry.
4. Detect pin candidates with grayscale thresholding, morphology, and connected components.
5. Convert candidates into normalized pin records relative to the IC body.
6. Match detected pins to a 16-pin-per-row reference layout.
7. Evaluate Missing, Pitch, Position, Width, Height, Area, and Shape candidates.

### CNN branch

1. Use matched reference slots to define stable Pin crop centers.
2. Extract a fixed `20 x 40` patch.
3. Pad to square, resize to `224 x 224`, normalize with ImageNet statistics.
4. Run a ResNet18 classifier exported to ONNX.
5. Use `P(defect) >= 0.50` as the prototype CNN NG condition.

### Fusion

```text
Final NG = Rule NG OR CNN NG
```

Missing pins are skipped by the CNN branch because no physical Pin patch exists at the missing slot; they remain the responsibility of the geometry branch.

## Validated 8-case regression

- Normal: OK
- Missing Top: NG
- Missing Bottom: NG
- Position Shift: NG
- Width: NG
- Height: NG
- Area: NG
- Shape/Bent synthetic case: NG

Result: `8/8 PASS` on the fixed controlled regression set.

## Supplemental experiments

### PatchCore-style anomaly detection

Uses pretrained ResNet18 intermediate features, a normal-feature memory bank, nearest-neighbor patch distances, image-level anomaly score, and heatmap visualization. The implementation intentionally uses random memory subsampling rather than the full paper's greedy coreset selection.

### YOLO

Runs pretrained YOLO11n inference and includes a small explicit IoU/NMS example.

### Open3D

Runs a synthetic point-cloud pipeline with voxel downsampling, statistical outlier removal, RANSAC plane segmentation, and DBSCAN object clustering.

## Evidence boundary

This repository demonstrates an end-to-end engineering prototype and controlled regression behavior. It does not establish production-level generalization, calibrated physical measurement, or industrial false-reject / escape rates.
