#!/usr/bin/env python3
"""
Batch processing script for Kaggle tipburn dataset using Grounded-SAM-2.
Processes 626 healthy strawberry leaf images to generate bounding boxes and segmentation masks.

Usage:
    python batch_process_kaggle.py --images_dir /path/to/images --output_dir /path/to/output

This script will:
1. Load Grounding DINO and SAM 2 models
2. Process each image with text prompt "strawberry leaf"
3. Generate bboxes and segmentation masks
4. Save results as JSON (one file per image)
5. Generate summary statistics

Output format (per image):
{
    "image_path": "path/to/image.jpg",
    "image_size": [width, height],
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],  // xyxy format
            "confidence": 0.85,
            "label": "strawberry leaf",
            "segmentation_rle": {...}  // RLE encoded mask
        },
        ...
    ]
}
"""

import os
import json
import torch
import numpy as np
import cv2
import pycocotools.mask as mask_util
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
import argparse
import time

# Hyperparameters
TEXT_PROMPT = "strawberry leaf."  # Note: lowercase + dot
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.30  # Lower threshold to catch more leaves
TEXT_THRESHOLD = 0.25
MULTIMASK_OUTPUT = False

def single_mask_to_rle(mask):
    """Convert binary mask to RLE format."""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def process_single_image(image_path, grounding_model, sam2_predictor, device, output_dir):
    """
    Process a single image and return detection results.

    Returns:
        dict: Detection results with bboxes, confidences, labels, and masks
    """
    try:
        # Load image
        image_source, image = load_image(str(image_path))
        h, w, _ = image_source.shape

        # Set image for SAM 2
        sam2_predictor.set_image(image_source)

        # Run Grounding DINO
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )

        # Check if any detections
        if len(boxes) == 0:
            return {
                "image_path": str(image_path),
                "image_size": [w, h],
                "detections": [],
                "status": "no_detections"
            }

        # Convert boxes to xyxy format
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Run SAM 2 segmentation
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=MULTIMASK_OUTPUT,
        )

        # Sample best mask if multiple
        if MULTIMASK_OUTPUT and masks.ndim == 4:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]

        # Ensure masks are (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Build detection results
        detections = []
        for i in range(len(input_boxes)):
            detection = {
                "bbox": input_boxes[i].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(confidences[i]),
                "label": labels[i],
                "segmentation_rle": single_mask_to_rle(masks[i])
            }
            detections.append(detection)

        result = {
            "image_path": str(image_path),
            "image_size": [w, h],
            "detections": detections,
            "status": "success"
        }

        # Save individual JSON result
        json_path = output_dir / "json" / f"{image_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return {
            "image_path": str(image_path),
            "image_size": [0, 0],
            "detections": [],
            "status": f"error: {str(e)}"
        }

def main():
    parser = argparse.ArgumentParser(description="Batch process images with Grounded-SAM-2")
    parser.add_argument("--images_dir", type=str, default="data/external/kaggle_tipburn/train/images",
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="grounded_sam2_results",
                        help="Output directory for results")
    parser.add_argument("--checkpoint_dir", type=str, default="Grounded-SAM-2",
                        help="Grounded-SAM-2 repo directory")
    args = parser.parse_args()

    # Setup paths
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(exist_ok=True)

    # Change to checkpoint directory
    os.chdir(checkpoint_dir)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Load models
    print("\nLoading SAM 2 model...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading Grounding DINO model...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    # Enable optimizations for Ampere GPUs (RTX 3000+, A100, etc.)
    if torch.cuda.is_available():
        torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
    image_files = sorted(image_files)

    print(f"\nFound {len(image_files)} images in {images_dir}")
    print(f"Text prompt: '{TEXT_PROMPT}'")
    print(f"Box threshold: {BOX_THRESHOLD}")
    print(f"Output directory: {output_dir}")
    print("\nStarting batch processing...")
    print("=" * 60)

    # Process all images
    results = []
    start_time = time.time()

    for image_path in tqdm(image_files, desc="Processing images"):
        result = process_single_image(
            image_path, grounding_model, sam2_predictor, device, output_dir
        )
        results.append(result)

    elapsed_time = time.time() - start_time

    # Generate summary statistics
    total_images = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    no_detections = sum(1 for r in results if r["status"] == "no_detections")
    errors = sum(1 for r in results if r["status"].startswith("error"))
    total_detections = sum(len(r["detections"]) for r in results)

    avg_detections = total_detections / total_images if total_images > 0 else 0
    avg_time_per_image = elapsed_time / total_images if total_images > 0 else 0

    summary = {
        "total_images": total_images,
        "successful": successful,
        "no_detections": no_detections,
        "errors": errors,
        "total_detections": total_detections,
        "avg_detections_per_image": round(avg_detections, 2),
        "total_time_seconds": round(elapsed_time, 2),
        "avg_time_per_image_seconds": round(avg_time_per_image, 2),
        "text_prompt": TEXT_PROMPT,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "device": device
    }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save all results
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images:              {total_images}")
    print(f"Successful:                {successful}")
    print(f"No detections:             {no_detections}")
    print(f"Errors:                    {errors}")
    print(f"Total detections:          {total_detections}")
    print(f"Avg detections per image:  {avg_detections:.2f}")
    print(f"Total time:                {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
    print(f"Avg time per image:        {avg_time_per_image:.2f}s")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"All results: {all_results_path}")
    print(f"Individual JSONs: {output_dir / 'json'}/")
    print("\nNext step: Download results and convert to YOLO format")

if __name__ == "__main__":
    main()
