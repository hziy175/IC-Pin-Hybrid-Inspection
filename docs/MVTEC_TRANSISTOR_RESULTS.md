# MVTec AD `transistor` Full-Test Evaluation

This repository includes a lightweight **PatchCore-style** anomaly-detection experiment on the real MVTec AD `transistor` category.

## Protocol

- Training side: `transistor/train/good`
- Normal training images: **213**
- Test side: full `transistor/test`
- Test images: **100**
  - `good`: 60
  - `bent_lead`: 10
  - `cut_lead`: 10
  - `damaged_case`: 10
  - `misplaced`: 10
- Feature extractor: ImageNet-pretrained ResNet18, frozen
- Features: `layer2 + layer3`
- Input size: `224 x 224`
- Memory bank: 166,992 extracted normal patch features, randomly subsampled to 10,000
- Image score: mean of the highest 1% nearest-neighbor patch distances
- Test labels are used only for evaluation, not for memory-bank construction.

## Results

| Metric | Result |
|---|---:|
| Image-level AUROC | **0.937083** |
| Image-level AP | **0.954614** |
| Pixel-level AUROC | **0.841665** |
| Pixel-level AP | **0.478352** |

Per-defect results, each defect type compared with all 60 `good` test images:

| Defect type | Image AUROC | Image AP | Pixel AUROC | Pixel AP |
|---|---:|---:|---:|---:|
| `bent_lead` | **0.996667** | **0.983333** | 0.849390 | 0.281278 |
| `cut_lead` | 0.953333 | 0.884483 | 0.625902 | 0.083706 |
| `damaged_case` | **0.998333** | **0.990909** | **0.987081** | **0.751699** |
| `misplaced` | 0.800000 | 0.827329 | 0.809685 | 0.763791 |

Score statistics:

| Category | N | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
| `good` | 60 | 0.685982 | 0.026049 | 0.637074 | 0.680441 | 0.742615 |
| `bent_lead` | 10 | 0.780471 | 0.033542 | 0.738993 | 0.766701 | 0.828430 |
| `cut_lead` | 10 | 0.748494 | 0.025364 | 0.696279 | 0.750781 | 0.787302 |
| `damaged_case` | 10 | 0.828197 | 0.046634 | 0.742207 | 0.840386 | 0.896902 |
| `misplaced` | 10 | 0.824193 | 0.109944 | 0.624079 | 0.857885 | 0.971054 |

## Interpretation

The full test shows strong image-level anomaly ranking overall, but the method does not perfectly separate every normal and abnormal sample. The maximum `good` score is 0.742615 while the minimum defect score is 0.624079.

`damaged_case` is the easiest category for the current representation at both image and pixel level. `misplaced` is the most difficult image-level category and shows high intra-class variation. `cut_lead` remains detectable at image level but is difficult to localize precisely after whole-image resizing to `224 x 224`.

## Scope and limitations

This experiment is **not an official PatchCore reproduction**. It uses a simplified feature-memory implementation with random memory-bank subsampling rather than the original greedy coreset selection. Pixel-level metrics are calculated after resizing images and masks to the network input size, so they should not be compared directly with official benchmark numbers.

The MVTec AD images are **not included in this repository**. Download the original dataset from MVTec and pass its parent directory with `--mvtec-root`.
