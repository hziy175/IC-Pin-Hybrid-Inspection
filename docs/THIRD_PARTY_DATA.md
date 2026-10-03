# Third-Party Data and Model Notes

## MVTec AD

The MVTec AD dataset is not redistributed in this repository.

Official page:

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

MVTec states that the dataset is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license and is not permitted for commercial use.

The full `transistor` evaluation script expects the official dataset to exist outside this repository, for example:

```text
D:\datasets\transistor\
├─ train\good\
├─ test\...
└─ ground_truth\...
```

Run with:

```powershell
python .\experiments\patchcore_transistor_full_eval.py --mvtec-root D:\datasets
```

## YOLO11n

The repository does not redistribute a YOLO11n weight file. `experiments/yolo_demo.py` requests the standard `yolo11n.pt` model through the installed Ultralytics package on first use. Users should follow the upstream package/model licensing terms.
