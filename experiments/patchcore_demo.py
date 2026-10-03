from pathlib import Path
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
ROOT_DIR = Path(__file__).resolve().parents[1]
NORMAL_DIR = ROOT_DIR / 'data' / 'pin_cls_raw' / 'normal'
DEFECT_DIR = ROOT_DIR / 'data' / 'pin_cls_raw' / 'defect_synthetic'
OUTPUT_DIR = ROOT_DIR / 'output' / 'experiments' / 'patchcore'
HEATMAP_DIR = OUTPUT_DIR / 'heatmaps'
CSV_PATH = OUTPUT_DIR / 'patchcore_scores.csv'
IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_MEMORY_PATCHES = 4096
SEED = 42
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def find_images(folder):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sorted([path for path in folder.iterdir() if path.suffix.lower() in extensions])

def pad_to_square_by_edge(image_bgr):
    h, w = image_bgr.shape[:2]
    if h == w:
        return image_bgr
    if h > w:
        difference = h - w
        left = difference // 2
        right = difference - left
        return cv2.copyMakeBorder(image_bgr, 0, 0, left, right, cv2.BORDER_REPLICATE)
    difference = w - h
    top = difference // 2
    bottom = difference - top
    return cv2.copyMakeBorder(image_bgr, top, bottom, 0, 0, cv2.BORDER_REPLICATE)

def preprocess_image(image_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    square_bgr = pad_to_square_by_edge(image_bgr)
    resized_bgr = cv2.resize(square_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    image_array = resized_rgb.astype(np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    tensor = torch.from_numpy(image_array).float()
    return (tensor, resized_bgr)

class ResNet18FeatureExtractor:

    def __init__(self, device):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = device
        self.features = {}
        self.model.layer2.register_forward_hook(self._save_layer2)
        self.model.layer3.register_forward_hook(self._save_layer3)

    def _save_layer2(self, module, inputs, output):
        self.features['layer2'] = output

    def _save_layer3(self, module, inputs, output):
        self.features['layer3'] = output

    @torch.inference_mode()
    def extract(self, batch):
        self.features = {}
        _ = self.model(batch.to(self.device))
        feature2 = self.features['layer2']
        feature3 = self.features['layer3']
        feature3 = F.interpolate(feature3, size=feature2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([feature2, feature3], dim=1)
        features = F.normalize(features, p=2, dim=1)
        return features

def feature_map_to_patch_vectors(features):
    vectors = features.permute(0, 2, 3, 1).contiguous()
    b, h, w, c = vectors.shape
    vectors = vectors.view(b, h * w, c)
    return (vectors, h, w)

def build_memory_bank(extractor, normal_paths):
    all_vectors = []
    for start in range(0, len(normal_paths), BATCH_SIZE):
        batch_paths = normal_paths[start:start + BATCH_SIZE]
        tensors = []
        for image_path in batch_paths:
            tensor, _ = preprocess_image(image_path)
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0)
        feature_map = extractor.extract(batch)
        vectors, _, _ = feature_map_to_patch_vectors(feature_map)
        all_vectors.append(vectors.reshape(-1, vectors.shape[-1]).cpu())
    memory_bank = torch.cat(all_vectors, dim=0)
    original_count = memory_bank.shape[0]
    if original_count > MAX_MEMORY_PATCHES:
        generator = torch.Generator()
        generator.manual_seed(SEED)
        indices = torch.randperm(original_count, generator=generator)[:MAX_MEMORY_PATCHES]
        memory_bank = memory_bank[indices]
    return (memory_bank, original_count)

@torch.inference_mode()
def compute_anomaly_map(extractor, memory_bank, image_path):
    tensor, resized_bgr = preprocess_image(image_path)
    feature_map = extractor.extract(tensor.unsqueeze(0))
    vectors, h, w = feature_map_to_patch_vectors(feature_map)
    vectors = vectors[0].cpu()
    distances = torch.cdist(vectors, memory_bank)
    nearest_distance = distances.min(dim=1).values
    patch_map = nearest_distance.view(1, 1, h, w)
    full_map = F.interpolate(patch_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)[0, 0]
    top_k = max(1, int(nearest_distance.numel() * 0.01))
    image_score = torch.topk(nearest_distance, k=top_k).values.mean().item()
    return (image_score, full_map.numpy(), resized_bgr)

def make_heatmap_overlay(image_bgr, anomaly_map, global_min, global_max):
    denominator = global_max - global_min
    if denominator < 1e-12:
        normalized = np.zeros_like(anomaly_map, dtype=np.float32)
    else:
        normalized = (anomaly_map - global_min) / denominator
    normalized = np.clip(normalized, 0.0, 1.0)
    heatmap_uint8 = (normalized * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap_bgr, 0.45, 0.0)
    return (heatmap_bgr, overlay)

def choose_normal_t08(normal_paths):
    candidates = [path for path in normal_paths if 'T08' in path.stem.upper()]
    if candidates:
        return candidates[0]
    return normal_paths[0]

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not NORMAL_DIR.exists():
        raise FileNotFoundError(f'Normal folder not found: {NORMAL_DIR}')
    if not DEFECT_DIR.exists():
        raise FileNotFoundError(f'Defect folder not found: {DEFECT_DIR}')
    normal_paths = find_images(NORMAL_DIR)
    defect_paths = find_images(DEFECT_DIR)
    if len(normal_paths) == 0:
        raise RuntimeError('No normal images found.')
    if len(defect_paths) == 0:
        raise RuntimeError('No defect images found.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('PATCHCORE ANOMALY DETECTION DEMO')
    print('=' * 80)
    print()
    print(f'Device = {device}')
    print(f'Normal training patches = {len(normal_paths)}')
    print(f'Synthetic defect patches = {len(defect_paths)}')
    extractor = ResNet18FeatureExtractor(device)
    print()
    print('Building normal feature memory bank...')
    memory_bank, original_count = build_memory_bank(extractor, normal_paths)
    print(f'Original normal patch features = {original_count}')
    print(f'Memory bank used = {memory_bank.shape[0]}')
    print(f'Feature dimension = {memory_bank.shape[1]}')
    normal_demo_path = choose_normal_t08(normal_paths)
    test_cases = [('normal', normal_demo_path)]
    for defect_path in defect_paths:
        test_cases.append(('synthetic_defect', defect_path))
    results = []
    print()
    print('Running anomaly inference...')
    for label, image_path in test_cases:
        score, anomaly_map, image_bgr = compute_anomaly_map(extractor, memory_bank, image_path)
        results.append({'label': label, 'path': image_path, 'score': score, 'anomaly_map': anomaly_map, 'image_bgr': image_bgr})
        print(f'{image_path.name:<40} label={label:<18} score={score:.6f}')
    global_min = min((float(result['anomaly_map'].min()) for result in results))
    global_max = max((float(result['anomaly_map'].max()) for result in results))
    rows = []
    for result in results:
        heatmap_bgr, overlay = make_heatmap_overlay(result['image_bgr'], result['anomaly_map'], global_min, global_max)
        stem = result['path'].stem
        heatmap_path = HEATMAP_DIR / f'{stem}_heatmap.jpg'
        overlay_path = HEATMAP_DIR / f'{stem}_overlay.jpg'
        cv2.imwrite(str(heatmap_path), heatmap_bgr)
        cv2.imwrite(str(overlay_path), overlay)
        rows.append({'image': result['path'].name, 'label': result['label'], 'anomaly_score': f"{result['score']:.6f}", 'heatmap': str(heatmap_path), 'overlay': str(overlay_path)})
    with CSV_PATH.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label', 'anomaly_score', 'heatmap', 'overlay'])
        writer.writeheader()
        writer.writerows(rows)
    normal_score = results[0]['score']
    defect_scores = [result['score'] for result in results[1:]]
    all_defects_higher = all((score > normal_score for score in defect_scores))
    print()
    print('=' * 80)
    print('PATCHCORE CHECK')
    print('=' * 80)
    print(f'Normal demo score = {normal_score:.6f}')
    print(f'All synthetic defect scores higher than normal = {all_defects_higher}')
    print()
    print('CSV saved to:')
    print(CSV_PATH)
    print()
    print('Heatmaps saved to:')
    print(HEATMAP_DIR)
    print()
    print('NOTE: This is a controlled demo. Do not interpret these scores as industrial accuracy.')
if __name__ == '__main__':
    main()
