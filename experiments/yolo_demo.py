from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output" / "experiments" / "yolo"
MODEL_NAME = "yolo11n.pt"
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.70


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def simple_nms(boxes, scores, iou_threshold):
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(int(current))

        if len(order) == 1:
            break

        remaining = order[1:]
        next_order = []

        for index in remaining:
            iou = box_iou(boxes[current], boxes[index])
            if iou <= iou_threshold:
                next_order.append(index)

        order = np.array(next_order, dtype=np.int64)

    return keep


def draw_toy_nms(boxes, scores, keep_indices):
    canvas_before = np.full((400, 600, 3), 255, dtype=np.uint8)
    canvas_after = canvas_before.copy()
    labels = ["A", "B", "C"]

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(canvas_before, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas_before, f"{labels[index]} {scores[index]:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for index in keep_indices:
        box = boxes[index]
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(canvas_after, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(canvas_after, f"{labels[index]} {scores[index]:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

    cv2.imwrite(str(OUTPUT_DIR / "toy_nms_before.jpg"), canvas_before)
    cv2.imwrite(str(OUTPUT_DIR / "toy_nms_after.jpg"), canvas_after)


def run_real_yolo_demo(image_path):
    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 88)
    print("YOLO OBJECT DETECTION DEMO")
    print("=" * 88)
    print()
    print(f"Model = {MODEL_NAME}")
    print(f"Input image = {image_path}")
    print(f"Device = {'cuda:0' if device == 0 else 'cpu'}")
    print(f"Confidence threshold = {CONF_THRESHOLD}")
    print(f"NMS IoU threshold = {NMS_IOU_THRESHOLD}")

    # Ultralytics will download the official pretrained weight on first use if needed.
    model = YOLO(MODEL_NAME)

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        iou=NMS_IOU_THRESHOLD,
        imgsz=640,
        device=device,
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()
    result_image_path = OUTPUT_DIR / "yolo_result.jpg"
    cv2.imwrite(str(result_image_path), annotated)

    print()
    print("-" * 88)
    print("YOLO DETECTIONS")
    print("-" * 88)

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        return

    xyxy = result.boxes.xyxy.cpu().numpy()
    confidence = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    print(f"{'ID':<5}{'Class':<15}{'Conf':<10}{'BBox [x1,y1,x2,y2]'}")
    print("-" * 88)

    for index in range(len(xyxy)):
        class_name = result.names[classes[index]]
        box = xyxy[index]
        print(f"{index:<5}{class_name:<15}{confidence[index]:<10.4f}[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

    print()
    print(f"Final detections after YOLO post-processing = {len(xyxy)}")
    print(f"Annotated result saved to: {result_image_path}")


def run_iou_nms_demo():
    boxes = np.array(
        [
            [100, 100, 300, 300],
            [120, 120, 310, 310],
            [350, 100, 500, 260],
        ],
        dtype=np.float32,
    )

    scores = np.array([0.95, 0.88, 0.80], dtype=np.float32)
    labels = ["A", "B", "C"]
    demo_threshold = 0.50

    iou_ab = box_iou(boxes[0], boxes[1])
    keep_indices = simple_nms(boxes, scores, demo_threshold)
    draw_toy_nms(boxes, scores, keep_indices)

    print()
    print("=" * 88)
    print("IoU + NMS TOY DEMO")
    print("=" * 88)
    print(f"IoU(A, B) = {iou_ab:.4f}")
    print(f"NMS threshold = {demo_threshold:.2f}")
    print(f"After NMS keep = {[labels[index] for index in keep_indices]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an image for pretrained YOLO11n inference.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_real_yolo_demo(Path(args.image).resolve())
    run_iou_nms_demo()


if __name__ == "__main__":
    main()
