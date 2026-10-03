from pathlib import Path
import argparse
import csv
import json
import math
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError as exc:
    raise SystemExit(
        "缺少 scikit-learn。\n"
        "请先执行：python -m pip install scikit-learn"
    ) from exc


# ============================================================
# 说明
# ============================================================
# 这是一个“PatchCore-style”完整 MVTec AD transistor 评测脚本。
#
# 它沿用之前快速实践版的核心机制：
# 1. 用 ImageNet 预训练 ResNet18 提取 layer2 + layer3 局部特征；
# 2. 用 transistor/train/good 建立正常特征库；
# 3. 默认从正常特征中随机保留最多 10000 个局部特征；
# 4. 测试特征到正常特征库做最近邻距离；
# 5. 最近邻距离形成异常热力图；
# 6. 图像级分数默认取最高 1% 局部异常距离的均值。
#
# 注意：
# - 这不是官方 PatchCore 完整复现；
# - 这里没有实现论文里的 greedy coreset；
# - 默认仍采用 224×224 整图缩放，便于与之前 10 张真实图实验保持一致；
# - 本脚本不使用 test 标签调阈值；
# - 核心结果使用无需阈值的 AUROC / AP。
#
# 输出：
# - 每张图的异常分数 CSV
# - 总体 image-level AUROC / AP
# - 每种缺陷相对 good 的 AUROC / AP
# - 可选 pixel-level AUROC / AP（使用官方 mask，统一到网络输入尺寸）
# - 每类分数统计
# - 热力图与叠加图
# ============================================================


DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_MEMORY_PATCHES = 10000
DEFAULT_TOP_PERCENT = 0.01
DEFAULT_BANK_CHUNK = 2048
DEFAULT_SEED = 42

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def list_images(folder):
    exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if not folder.exists():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in exts
        ]
    )


def collect_test_samples(transistor_root):
    test_root = transistor_root / "test"

    if not test_root.exists():
        raise FileNotFoundError(
            f"未找到 test 目录：{test_root}"
        )

    samples = []

    category_dirs = sorted(
        [
            path
            for path in test_root.iterdir()
            if path.is_dir()
        ],
        key=lambda p: (
            p.name != "good",
            p.name
        )
    )

    for category_dir in category_dirs:
        category = category_dir.name
        label = 0 if category == "good" else 1

        for image_path in list_images(
            category_dir
        ):
            samples.append(
                {
                    "image_path": image_path,
                    "category": category,
                    "label": label,
                }
            )

    if not samples:
        raise RuntimeError(
            f"test 目录中没有找到图片：{test_root}"
        )

    return samples


def resize_with_stretch(image_bgr, image_size):
    resized = cv2.resize(
        image_bgr,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR
    )

    return resized


def resize_with_padding(image_bgr, image_size):
    h, w = image_bgr.shape[:2]

    scale = min(
        image_size / w,
        image_size / h
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        image_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (image_size, image_size, 3),
        dtype=np.uint8
    )

    top = (
        image_size - new_h
    ) // 2

    left = (
        image_size - new_w
    ) // 2

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = resized

    return canvas


def preprocess_image(
    image_path,
    image_size,
    preprocess_mode
):
    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise FileNotFoundError(
            f"无法读取图片：{image_path}"
        )

    if preprocess_mode == "stretch":
        network_bgr = resize_with_stretch(
            image_bgr,
            image_size
        )

    elif preprocess_mode == "pad":
        network_bgr = resize_with_padding(
            image_bgr,
            image_size
        )

    else:
        raise ValueError(
            f"未知预处理方式：{preprocess_mode}"
        )

    rgb = cv2.cvtColor(
        network_bgr,
        cv2.COLOR_BGR2RGB
    )

    arr = (
        rgb.astype(np.float32)
        / 255.0
    )

    arr = (
        arr - IMAGENET_MEAN
    ) / IMAGENET_STD

    arr = np.transpose(
        arr,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        arr
    ).float()

    return (
        tensor,
        network_bgr,
    )


def preprocess_mask(
    mask_path,
    image_size,
    preprocess_mode
):
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            f"无法读取 mask：{mask_path}"
        )

    if preprocess_mode == "stretch":
        resized = cv2.resize(
            mask,
            (image_size, image_size),
            interpolation=cv2.INTER_NEAREST
        )

    elif preprocess_mode == "pad":
        h, w = mask.shape[:2]

        scale = min(
            image_size / w,
            image_size / h
        )

        new_w = max(
            1,
            int(round(w * scale))
        )

        new_h = max(
            1,
            int(round(h * scale))
        )

        small = cv2.resize(
            mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        resized = np.zeros(
            (image_size, image_size),
            dtype=np.uint8
        )

        top = (
            image_size - new_h
        ) // 2

        left = (
            image_size - new_w
        ) // 2

        resized[
            top:top + new_h,
            left:left + new_w
        ] = small

    else:
        raise ValueError(
            f"未知预处理方式：{preprocess_mode}"
        )

    binary = (
        resized > 0
    ).astype(np.uint8)

    return binary


def resolve_mask_path(
    transistor_root,
    category,
    image_path
):
    if category == "good":
        return None

    ground_truth_dir = (
        transistor_root
        / "ground_truth"
        / category
    )

    candidates = [
        ground_truth_dir
        / f"{image_path.stem}_mask.png",
        ground_truth_dir
        / f"{image_path.stem}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


class ResNet18FeatureExtractor:
    def __init__(self, device):
        self.device = device

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        ).to(device)

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.features = {}

        self.model.layer2.register_forward_hook(
            self._save_layer2
        )

        self.model.layer3.register_forward_hook(
            self._save_layer3
        )

    def _save_layer2(
        self,
        module,
        inputs,
        output
    ):
        self.features["layer2"] = output

    def _save_layer3(
        self,
        module,
        inputs,
        output
    ):
        self.features["layer3"] = output

    @torch.inference_mode()
    def extract(self, batch):
        self.features = {}

        _ = self.model(
            batch.to(self.device)
        )

        f2 = self.features["layer2"]
        f3 = self.features["layer3"]

        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        features = torch.cat(
            [f2, f3],
            dim=1
        )

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

        return features


def feature_map_to_vectors(features):
    vectors = features.permute(
        0,
        2,
        3,
        1
    ).contiguous()

    batch_size, h, w, channels = (
        vectors.shape
    )

    vectors = vectors.view(
        batch_size,
        h * w,
        channels
    )

    return (
        vectors,
        h,
        w,
    )


def build_memory_bank(
    extractor,
    normal_train_paths,
    image_size,
    preprocess_mode,
    batch_size,
    max_memory_patches,
    seed
):
    chunks = []

    total_batches = math.ceil(
        len(normal_train_paths)
        / batch_size
    )

    for batch_index, start in enumerate(
        range(
            0,
            len(normal_train_paths),
            batch_size
        ),
        start=1
    ):
        batch_paths = (
            normal_train_paths[
                start:start + batch_size
            ]
        )

        tensors = [
            preprocess_image(
                path,
                image_size,
                preprocess_mode
            )[0]
            for path in batch_paths
        ]

        batch = torch.stack(
            tensors,
            dim=0
        )

        features = extractor.extract(
            batch
        )

        vectors, _, _ = (
            feature_map_to_vectors(
                features
            )
        )

        chunks.append(
            vectors.reshape(
                -1,
                vectors.shape[-1]
            ).cpu()
        )

        print(
            f"\rBuilding memory bank "
            f"[{batch_index}/{total_batches}]",
            end="",
            flush=True
        )

    print()

    bank = torch.cat(
        chunks,
        dim=0
    )

    original_count = (
        bank.shape[0]
    )

    if (
        max_memory_patches > 0
        and original_count
        > max_memory_patches
    ):
        generator = (
            torch.Generator()
            .manual_seed(seed)
        )

        indices = torch.randperm(
            original_count,
            generator=generator
        )[:max_memory_patches]

        bank = bank[indices]

    bank = bank.to(
        extractor.device
    )

    return (
        bank,
        original_count,
    )


@torch.inference_mode()
def nearest_patch_distances(
    vectors,
    memory_bank,
    bank_chunk_size
):
    best = torch.full(
        (vectors.shape[0],),
        float("inf"),
        device=vectors.device
    )

    for start in range(
        0,
        memory_bank.shape[0],
        bank_chunk_size
    ):
        bank_chunk = (
            memory_bank[
                start:start
                + bank_chunk_size
            ]
        )

        distances = torch.cdist(
            vectors,
            bank_chunk
        )

        chunk_best = (
            distances
            .min(dim=1)
            .values
        )

        best = torch.minimum(
            best,
            chunk_best
        )

    return best


@torch.inference_mode()
def anomaly_score_and_map(
    extractor,
    memory_bank,
    image_path,
    image_size,
    preprocess_mode,
    top_percent,
    bank_chunk_size
):
    (
        tensor,
        network_bgr,
    ) = preprocess_image(
        image_path,
        image_size,
        preprocess_mode
    )

    features = extractor.extract(
        tensor.unsqueeze(0)
    )

    (
        vectors,
        feature_h,
        feature_w,
    ) = feature_map_to_vectors(
        features
    )

    vectors = vectors[0]

    nearest = (
        nearest_patch_distances(
            vectors,
            memory_bank,
            bank_chunk_size
        )
    )

    patch_map = nearest.view(
        1,
        1,
        feature_h,
        feature_w
    )

    full_map = F.interpolate(
        patch_map,
        size=(
            image_size,
            image_size
        ),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    top_k = max(
        1,
        int(
            round(
                nearest.numel()
                * top_percent
            )
        )
    )

    top_k = min(
        top_k,
        nearest.numel()
    )

    image_score = (
        torch.topk(
            nearest,
            k=top_k
        )
        .values
        .mean()
        .item()
    )

    return (
        image_score,
        full_map.detach().cpu().numpy(),
        network_bgr,
    )


def normalize_map(anomaly_map):
    minimum = float(
        anomaly_map.min()
    )

    maximum = float(
        anomaly_map.max()
    )

    if (
        maximum - minimum
        < 1e-12
    ):
        return np.zeros_like(
            anomaly_map,
            dtype=np.float32
        )

    normalized = (
        anomaly_map - minimum
    ) / (
        maximum - minimum
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


def save_visualizations(
    image_bgr,
    anomaly_map,
    overlay_path,
    heatmap_path
):
    normalized = normalize_map(
        anomaly_map
    )

    heat_uint8 = (
        normalized * 255
    ).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image_bgr,
        0.55,
        heat_bgr,
        0.45,
        0.0
    )

    cv2.imwrite(
        str(heatmap_path),
        heat_bgr
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )


def safe_roc_auc(
    y_true,
    y_score
):
    y_true = np.asarray(
        y_true,
        dtype=np.uint8
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float32
    )

    if len(
        np.unique(y_true)
    ) < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            y_score
        )
    )


def safe_average_precision(
    y_true,
    y_score
):
    y_true = np.asarray(
        y_true,
        dtype=np.uint8
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float32
    )

    if y_true.sum() == 0:
        return None

    return float(
        average_precision_score(
            y_true,
            y_score
        )
    )


def format_metric(value):
    if value is None:
        return "N/A"

    return f"{value:.6f}"


def calculate_score_stats(rows):
    categories = sorted(
        set(
            row["category"]
            for row in rows
        ),
        key=lambda x: (
            x != "good",
            x
        )
    )

    output = []

    for category in categories:
        scores = np.array(
            [
                row["anomaly_score"]
                for row in rows
                if row["category"]
                == category
            ],
            dtype=np.float64
        )

        output.append(
            {
                "category": category,
                "count": int(
                    len(scores)
                ),
                "mean": float(
                    scores.mean()
                ),
                "std": float(
                    scores.std()
                ),
                "min": float(
                    scores.min()
                ),
                "median": float(
                    np.median(scores)
                ),
                "max": float(
                    scores.max()
                ),
            }
        )

    return output


def calculate_category_metrics(rows):
    categories = sorted(
        {
            row["category"]
            for row in rows
            if row["category"]
            != "good"
        }
    )

    output = []

    for category in categories:
        selected = [
            row
            for row in rows
            if row["category"]
            in {
                "good",
                category
            }
        ]

        y_true = [
            0
            if row["category"]
            == "good"
            else 1
            for row in selected
        ]

        y_score = [
            row["anomaly_score"]
            for row in selected
        ]

        output.append(
            {
                "category": category,
                "good_count": sum(
                    1
                    for row
                    in selected
                    if row["category"]
                    == "good"
                ),
                "defect_count": sum(
                    1
                    for row
                    in selected
                    if row["category"]
                    == category
                ),
                "image_auroc": (
                    safe_roc_auc(
                        y_true,
                        y_score
                    )
                ),
                "image_ap": (
                    safe_average_precision(
                        y_true,
                        y_score
                    )
                ),
            }
        )

    return output


def write_rows_csv(
    output_path,
    rows
):
    if not rows:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "PatchCore-style full MVTec AD "
            "transistor evaluation."
        )
    )

    parser.add_argument(
        "--mvtec-root",
        required=True,
        help=(
            r"MVTec AD 根目录，例如 "
            r"D:\datasets"
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "结果目录。默认保存到脚本旁的 "
            "outputs_full_transistor"
        )
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=(
            "网络输入尺寸，默认 224"
        )
    )

    parser.add_argument(
        "--preprocess",
        choices=[
            "stretch",
            "pad",
        ],
        default="stretch",
        help=(
            "stretch：直接拉伸为正方形；"
            "pad：保持比例后补边。"
            "默认 stretch，与之前 real-10 实验一致。"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "构建正常特征库时的批大小，默认 8"
        )
    )

    parser.add_argument(
        "--max-memory-patches",
        type=int,
        default=DEFAULT_MAX_MEMORY_PATCHES,
        help=(
            "正常特征库最大局部特征数，默认 10000。"
            "设为 0 表示全部保留。"
        )
    )

    parser.add_argument(
        "--top-percent",
        type=float,
        default=DEFAULT_TOP_PERCENT,
        help=(
            "图像级分数使用最高多少比例的局部异常值。"
            "默认 0.01，即最高 1%%。"
        )
    )

    parser.add_argument(
        "--bank-chunk",
        type=int,
        default=DEFAULT_BANK_CHUNK,
        help=(
            "最近邻计算时每次使用多少正常特征，"
            "默认 2048，用于控制显存。"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "正常特征随机抽样种子，默认 42"
        )
    )

    parser.add_argument(
        "--no-pixel-metrics",
        action="store_true",
        help=(
            "不计算 pixel-level AUROC / AP"
        )
    )

    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help=(
            "不保存热力图和叠加图"
        )
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.top_percent
        <= 1.0
    ):
        raise ValueError(
            "--top-percent 必须在 (0, 1] 范围内"
        )

    if args.image_size < 64:
        raise ValueError(
            "--image-size 不能小于 64"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size 必须 >= 1"
        )

    if args.bank_chunk < 1:
        raise ValueError(
            "--bank-chunk 必须 >= 1"
        )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    mvtec_root = (
        Path(args.mvtec_root)
        .resolve()
    )

    transistor_root = (
        mvtec_root
        / "transistor"
    )

    train_good_dir = (
        transistor_root
        / "train"
        / "good"
    )

    if not train_good_dir.exists():
        raise FileNotFoundError(
            "未找到 transistor/train/good。\n"
            f"当前查找路径：{train_good_dir}\n"
            "如果你的数据目录是 "
            r"D:\datasets\transistor，"
            "那么 --mvtec-root 应传 D:\\datasets"
        )

    if args.output_dir is None:
        output_dir = (
            script_dir
            / "outputs_full_transistor"
        )
    else:
        output_dir = (
            Path(args.output_dir)
            .resolve()
        )

    heatmap_dir = (
        output_dir
        / "heatmaps"
    )

    overlay_dir = (
        output_dir
        / "overlays"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not args.no_visuals:
        heatmap_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        overlay_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    normal_train_paths = (
        list_images(
            train_good_dir
        )
    )

    test_samples = (
        collect_test_samples(
            transistor_root
        )
    )

    print(
        "=" * 96
    )

    print(
        "MVTec AD TRANSISTOR "
        "FULL PATCHCORE-STYLE EVALUATION"
    )

    print(
        "=" * 96
    )

    print(
        f"Device = {device}"
    )

    print(
        f"Transistor root = "
        f"{transistor_root}"
    )

    print(
        f"Normal train images = "
        f"{len(normal_train_paths)}"
    )

    print(
        f"Full test images = "
        f"{len(test_samples)}"
    )

    print(
        f"Image size = "
        f"{args.image_size}"
    )

    print(
        f"Preprocess = "
        f"{args.preprocess}"
    )

    print(
        f"Top percent = "
        f"{args.top_percent:.4f}"
    )

    print(
        f"Max memory patches = "
        f"{args.max_memory_patches}"
    )

    categories = {}

    for sample in test_samples:
        categories[
            sample["category"]
        ] = (
            categories.get(
                sample["category"],
                0
            )
            + 1
        )

    print(
        "\nTest category counts:"
    )

    for category, count in (
        sorted(
            categories.items(),
            key=lambda item: (
                item[0] != "good",
                item[0]
            )
        )
    ):
        print(
            f"  {category:<18} "
            f"{count}"
        )

    extractor = (
        ResNet18FeatureExtractor(
            device
        )
    )

    print(
        "\nBuilding normal memory bank ..."
    )

    (
        memory_bank,
        original_patch_count,
    ) = build_memory_bank(
        extractor=extractor,
        normal_train_paths=normal_train_paths,
        image_size=args.image_size,
        preprocess_mode=args.preprocess,
        batch_size=args.batch_size,
        max_memory_patches=args.max_memory_patches,
        seed=args.seed
    )

    print(
        f"Original normal patch features = "
        f"{original_patch_count}"
    )

    print(
        f"Memory bank used = "
        f"{memory_bank.shape[0]}"
    )

    print(
        f"Feature dimension = "
        f"{memory_bank.shape[1]}"
    )

    print(
        "\nEvaluating full test set ..."
    )

    result_rows = []

    all_pixel_labels = []
    all_pixel_scores = []

    per_category_pixel_labels = {}
    per_category_pixel_scores = {}

    total_test = len(
        test_samples
    )

    missing_masks = []

    for index, sample in enumerate(
        test_samples,
        start=1
    ):
        image_path = (
            sample["image_path"]
        )

        category = (
            sample["category"]
        )

        label = (
            sample["label"]
        )

        (
            score,
            anomaly_map,
            network_bgr,
        ) = anomaly_score_and_map(
            extractor=extractor,
            memory_bank=memory_bank,
            image_path=image_path,
            image_size=args.image_size,
            preprocess_mode=args.preprocess,
            top_percent=args.top_percent,
            bank_chunk_size=args.bank_chunk
        )

        mask_path = resolve_mask_path(
            transistor_root,
            category,
            image_path
        )

        if not args.no_visuals:
            category_heatmap_dir = (
                heatmap_dir
                / category
            )

            category_overlay_dir = (
                overlay_dir
                / category
            )

            category_heatmap_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            category_overlay_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            save_visualizations(
                image_bgr=network_bgr,
                anomaly_map=anomaly_map,
                overlay_path=(
                    category_overlay_dir
                    / f"{image_path.stem}_overlay.jpg"
                ),
                heatmap_path=(
                    category_heatmap_dir
                    / f"{image_path.stem}_heatmap.jpg"
                )
            )

        if not args.no_pixel_metrics:
            if category == "good":
                gt_mask = np.zeros(
                    (
                        args.image_size,
                        args.image_size
                    ),
                    dtype=np.uint8
                )

            elif (
                mask_path is not None
                and mask_path.exists()
            ):
                gt_mask = preprocess_mask(
                    mask_path=mask_path,
                    image_size=args.image_size,
                    preprocess_mode=args.preprocess
                )

            else:
                gt_mask = None

                missing_masks.append(
                    str(image_path)
                )

            if gt_mask is not None:
                pixel_labels = (
                    gt_mask
                    .reshape(-1)
                    .astype(np.uint8)
                )

                pixel_scores = (
                    anomaly_map
                    .reshape(-1)
                    .astype(np.float32)
                )

                all_pixel_labels.append(
                    pixel_labels
                )

                all_pixel_scores.append(
                    pixel_scores
                )

                if category != "good":
                    per_category_pixel_labels.setdefault(
                        category,
                        []
                    ).append(
                        pixel_labels
                    )

                    per_category_pixel_scores.setdefault(
                        category,
                        []
                    ).append(
                        pixel_scores
                    )

        result_rows.append(
            {
                "image": image_path.name,
                "category": category,
                "label": label,
                "anomaly_score": float(
                    score
                ),
                "mask_found": (
                    category == "good"
                    or mask_path is not None
                ),
            }
        )

        print(
            f"[{index:03d}/{total_test:03d}] "
            f"{category:<18} "
            f"{image_path.name:<16} "
            f"score={score:.6f}"
        )

    image_labels = [
        row["label"]
        for row in result_rows
    ]

    image_scores = [
        row["anomaly_score"]
        for row in result_rows
    ]

    overall_image_auroc = (
        safe_roc_auc(
            image_labels,
            image_scores
        )
    )

    overall_image_ap = (
        safe_average_precision(
            image_labels,
            image_scores
        )
    )

    good_scores = np.array(
        [
            row["anomaly_score"]
            for row in result_rows
            if row["label"] == 0
        ],
        dtype=np.float64
    )

    defect_scores = np.array(
        [
            row["anomaly_score"]
            for row in result_rows
            if row["label"] == 1
        ],
        dtype=np.float64
    )

    separation_flag = None

    if (
        len(good_scores) > 0
        and len(defect_scores) > 0
    ):
        separation_flag = bool(
            defect_scores.min()
            > good_scores.max()
        )

    overall_pixel_auroc = None
    overall_pixel_ap = None

    if (
        not args.no_pixel_metrics
        and all_pixel_labels
    ):
        pixel_labels = np.concatenate(
            all_pixel_labels
        )

        pixel_scores = np.concatenate(
            all_pixel_scores
        )

        overall_pixel_auroc = (
            safe_roc_auc(
                pixel_labels,
                pixel_scores
            )
        )

        overall_pixel_ap = (
            safe_average_precision(
                pixel_labels,
                pixel_scores
            )
        )

    category_metrics = (
        calculate_category_metrics(
            result_rows
        )
    )

    pixel_category_metrics = []

    if not args.no_pixel_metrics:
        for category in sorted(
            per_category_pixel_labels.keys()
        ):
            labels = np.concatenate(
                per_category_pixel_labels[
                    category
                ]
            )

            scores = np.concatenate(
                per_category_pixel_scores[
                    category
                ]
            )

            pixel_category_metrics.append(
                {
                    "category": category,
                    "pixel_auroc": (
                        safe_roc_auc(
                            labels,
                            scores
                        )
                    ),
                    "pixel_ap": (
                        safe_average_precision(
                            labels,
                            scores
                        )
                    ),
                    "pixel_count": int(
                        len(labels)
                    ),
                    "positive_pixel_count": int(
                        labels.sum()
                    ),
                }
            )

    pixel_metric_map = {
        row["category"]: row
        for row in pixel_category_metrics
    }

    for row in category_metrics:
        pixel_row = (
            pixel_metric_map.get(
                row["category"]
            )
        )

        row["pixel_auroc"] = (
            None
            if pixel_row is None
            else pixel_row[
                "pixel_auroc"
            ]
        )

        row["pixel_ap"] = (
            None
            if pixel_row is None
            else pixel_row[
                "pixel_ap"
            ]
        )

    score_stats = (
        calculate_score_stats(
            result_rows
        )
    )

    result_csv_rows = []

    for row in result_rows:
        result_csv_rows.append(
            {
                "image": row["image"],
                "category": row["category"],
                "label": row["label"],
                "anomaly_score": (
                    f"{row['anomaly_score']:.8f}"
                ),
                "mask_found": row["mask_found"],
            }
        )

    metric_csv_rows = []

    for row in category_metrics:
        metric_csv_rows.append(
            {
                "category": row["category"],
                "good_count": row["good_count"],
                "defect_count": row["defect_count"],
                "image_auroc": (
                    ""
                    if row["image_auroc"]
                    is None
                    else (
                        f"{row['image_auroc']:.8f}"
                    )
                ),
                "image_ap": (
                    ""
                    if row["image_ap"]
                    is None
                    else (
                        f"{row['image_ap']:.8f}"
                    )
                ),
                "pixel_auroc": (
                    ""
                    if row["pixel_auroc"]
                    is None
                    else (
                        f"{row['pixel_auroc']:.8f}"
                    )
                ),
                "pixel_ap": (
                    ""
                    if row["pixel_ap"]
                    is None
                    else (
                        f"{row['pixel_ap']:.8f}"
                    )
                ),
            }
        )

    stat_csv_rows = []

    for row in score_stats:
        stat_csv_rows.append(
            {
                "category": row["category"],
                "count": row["count"],
                "mean": f"{row['mean']:.8f}",
                "std": f"{row['std']:.8f}",
                "min": f"{row['min']:.8f}",
                "median": f"{row['median']:.8f}",
                "max": f"{row['max']:.8f}",
            }
        )

    write_rows_csv(
        output_dir
        / "image_scores.csv",
        result_csv_rows
    )

    write_rows_csv(
        output_dir
        / "per_category_metrics.csv",
        metric_csv_rows
    )

    write_rows_csv(
        output_dir
        / "score_statistics.csv",
        stat_csv_rows
    )

    summary = {
        "method": (
            "PatchCore-style ResNet18 "
            "layer2+layer3 nearest-neighbor"
        ),
        "official_patchcore_reproduction": False,
        "device": str(device),
        "image_size": args.image_size,
        "preprocess": args.preprocess,
        "normal_train_images": len(
            normal_train_paths
        ),
        "test_images": len(
            test_samples
        ),
        "original_normal_patch_features": int(
            original_patch_count
        ),
        "memory_bank_used": int(
            memory_bank.shape[0]
        ),
        "feature_dimension": int(
            memory_bank.shape[1]
        ),
        "top_percent": args.top_percent,
        "overall_image_auroc": (
            overall_image_auroc
        ),
        "overall_image_ap": (
            overall_image_ap
        ),
        "overall_pixel_auroc": (
            overall_pixel_auroc
        ),
        "overall_pixel_ap": (
            overall_pixel_ap
        ),
        "min_defect_score": (
            None
            if len(defect_scores) == 0
            else float(
                defect_scores.min()
            )
        ),
        "max_good_score": (
            None
            if len(good_scores) == 0
            else float(
                good_scores.max()
            )
        ),
        "all_defects_above_all_good": (
            separation_flag
        ),
        "missing_masks": missing_masks,
    }

    with (
        output_dir
        / "metrics_summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2
        )

    report_lines = []

    report_lines.append(
        "=" * 96
    )

    report_lines.append(
        "FULL TRANSISTOR EVALUATION SUMMARY"
    )

    report_lines.append(
        "=" * 96
    )

    report_lines.append(
        f"Image-level AUROC = "
        f"{format_metric(overall_image_auroc)}"
    )

    report_lines.append(
        f"Image-level AP    = "
        f"{format_metric(overall_image_ap)}"
    )

    if not args.no_pixel_metrics:
        report_lines.append(
            f"Pixel-level AUROC = "
            f"{format_metric(overall_pixel_auroc)}"
        )

        report_lines.append(
            f"Pixel-level AP    = "
            f"{format_metric(overall_pixel_ap)}"
        )

    if (
        len(good_scores) > 0
        and len(defect_scores) > 0
    ):
        report_lines.append(
            f"Mean good score   = "
            f"{good_scores.mean():.6f}"
        )

        report_lines.append(
            f"Mean defect score = "
            f"{defect_scores.mean():.6f}"
        )

        report_lines.append(
            f"Max good score    = "
            f"{good_scores.max():.6f}"
        )

        report_lines.append(
            f"Min defect score  = "
            f"{defect_scores.min():.6f}"
        )

        report_lines.append(
            "All defects > all good = "
            f"{separation_flag}"
        )

    report_lines.append(
        "\nPer-defect metrics "
        "(each defect type vs all good images):"
    )

    for row in category_metrics:
        line = (
            f"{row['category']:<18} "
            f"image_AUROC="
            f"{format_metric(row['image_auroc'])} "
            f"image_AP="
            f"{format_metric(row['image_ap'])}"
        )

        if not args.no_pixel_metrics:
            line += (
                f" pixel_AUROC="
                f"{format_metric(row['pixel_auroc'])}"
                f" pixel_AP="
                f"{format_metric(row['pixel_ap'])}"
            )

        report_lines.append(
            line
        )

    report_lines.append(
        "\nScore statistics:"
    )

    for row in score_stats:
        report_lines.append(
            f"{row['category']:<18} "
            f"n={row['count']:<3d} "
            f"mean={row['mean']:.6f} "
            f"std={row['std']:.6f} "
            f"min={row['min']:.6f} "
            f"median={row['median']:.6f} "
            f"max={row['max']:.6f}"
        )

    if missing_masks:
        report_lines.append(
            "\nWARNING: "
            f"{len(missing_masks)} "
            "张异常图没有找到 mask，"
            "因此未纳入 pixel-level 指标。"
        )

    report_lines.append(
        "\nImportant:"
    )

    report_lines.append(
        "- AUROC / AP 不需要人为选测试集阈值。"
    )

    report_lines.append(
        "- 不要根据完整 test 集结果再反复调参后，"
        "把同一 test 集当作独立证明。"
    )

    report_lines.append(
        "- 当前方法是 PatchCore-style 快速实现，"
        "不是官方 PatchCore 完整复现。"
    )

    report_lines.append(
        "- pixel-level 指标在统一后的网络输入尺寸上计算，"
        "不是官方原分辨率 benchmark 数字。"
    )

    report_text = "\n".join(
        report_lines
    )

    print(
        "\n"
        + report_text
    )

    (
        output_dir
        / "evaluation_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8"
    )

    print(
        "\nSaved:"
    )

    print(
        output_dir
        / "image_scores.csv"
    )

    print(
        output_dir
        / "per_category_metrics.csv"
    )

    print(
        output_dir
        / "score_statistics.csv"
    )

    print(
        output_dir
        / "metrics_summary.json"
    )

    print(
        output_dir
        / "evaluation_report.txt"
    )

    if not args.no_visuals:
        print(
            heatmap_dir
        )

        print(
            overlay_dir
        )


if __name__ == "__main__":
    main()
