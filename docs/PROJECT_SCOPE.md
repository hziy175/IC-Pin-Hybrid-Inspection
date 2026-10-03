# Project Scope

## Main inspection system

The main project is an offline IC-pin inspection prototype with two complementary branches.

### Geometry branch

1. Detect the IC body by thresholding and contour filtering.
2. Estimate orientation with `minAreaRect` and deskew the image.
3. Generate top and bottom pin ROIs from body geometry.
4. Detect pin candidates using grayscale thresholding, morphology, and connected components.
5. Represent candidate coordinates relative to the IC body.
6. Match detected pins to the 16-pin-per-row reference layout.
7. Evaluate Missing, Pitch, Position, Width, Height, Area, and Shape abnormalities.

### CNN branch

1. Use matched reference slots to define stable pin crop centers.
2. Extract fixed `20 x 40` patches.
3. Pad to square and resize to `224 x 224`.
4. Run a ResNet18 classifier exported to ONNX.
5. Use `P(defect) >= 0.50` as the prototype CNN NG condition.

### Fusion

```text
Final NG = Rule NG OR CNN NG
```

Missing pins are handled by the geometry branch because no physical pin patch exists for CNN classification.

## Validated controlled regression

The fixed eight-case regression set contains one normal image and seven controlled abnormal cases covering missing pins, position shift, width, height, area, and shape changes.

Expected result:

```text
Regression = 8/8 PASS
Overall = PASS
```

This result was re-validated in the user's Windows environment using ONNX Runtime 1.30.0.

## Supplemental experiments

- **PatchCore-style anomaly detection**: synthetic local-pin mechanism demo plus a full real MVTec AD `transistor` evaluation.
- **YOLO11n**: pretrained object-detection inference plus an explicit IoU/NMS demonstration.
- **Open3D**: synthetic point-cloud processing with voxel downsampling, statistical outlier removal, RANSAC plane segmentation, and DBSCAN clustering.

## Evidence boundary

This repository demonstrates an offline engineering prototype and controlled/benchmark experimentation. It does not establish production-level generalization, calibrated physical measurement in millimeters, line-speed throughput, or industrial false-reject / escape rates.
