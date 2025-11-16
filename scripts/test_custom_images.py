#!/usr/bin/env python3
"""
Test trained model on custom images.
"""
from pathlib import Path
from ultralytics import YOLO

def test_custom_images(model_path: str, images_dir: str, output_dir: str = "runs/detect/custom_test"):
    """
    Run inference on custom images.

    Args:
        model_path: Path to trained model
        images_dir: Directory containing test images
        output_dir: Output directory for results
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))

    print(f"\nFound {len(image_files)} images in {images_dir}")
    print("Running inference...\n")

    # Run inference
    results = model.predict(
        source=str(images_dir),
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name="results",
        exist_ok=True,
        conf=0.25,  # Confidence threshold
        iou=0.45,   # NMS IoU threshold
    )

    # Print results for each image
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)

    for i, result in enumerate(results):
        img_name = image_files[i].name
        boxes = result.boxes

        print(f"\n{img_name}:")
        print(f"  Found {len(boxes)} detections")

        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"    - {class_name}: {conf:.2%} confidence")
        else:
            print("    No diseases detected (healthy or low confidence)")

    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}/results")
    print("="*60 + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model on custom images")
    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/raw/test/images",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detect/custom_test",
        help="Output directory for results"
    )

    args = parser.parse_args()

    test_custom_images(args.model, args.images, args.output)
