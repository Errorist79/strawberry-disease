#!/usr/bin/env python3
"""
Prepare PlantVillage dataset for CVAT import

This script:
1. Creates a list of images with invalid bboxes (priority for annotation)
2. Prepares import-ready directory structure
3. Generates task creation instructions
"""

import json
from pathlib import Path
from collections import defaultdict

def load_invalid_bbox_list(log_file):
    """Parse the invalid bbox log file"""
    invalid_images = defaultdict(list)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_split = None
    for line in lines:
        line = line.strip()

        # Detect split section
        if 'TRAIN Split' in line:
            current_split = 'train'
        elif 'VALID Split' in line or 'VAL Split' in line:
            current_split = 'valid'
        elif 'TEST Split' in line:
            current_split = 'test'

        # Extract image names
        if line.startswith('Image:'):
            image_name = line.replace('Image:', '').strip()
            if current_split:
                invalid_images[current_split].append(image_name)

    return invalid_images


def main():
    project_root = Path("/Users/errorist/Documents/personal-projects/strawberry-disease")
    plantvillage_root = project_root / "data/external/plantvillage_strawberry"
    invalid_bbox_log = project_root / "logs/plantvillage_invalid_bboxes.txt"
    output_dir = project_root / "cvat/import_data"

    print("=" * 70)
    print("PlantVillage CVAT Import Preparation")
    print("=" * 70)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load invalid bbox list
    print("📋 Loading invalid bbox list...")
    invalid_images = load_invalid_bbox_list(invalid_bbox_log)

    total_invalid = sum(len(imgs) for imgs in invalid_images.values())
    print(f"   Found {total_invalid} images with invalid bboxes:")
    for split, images in invalid_images.items():
        print(f"     {split}: {len(images)} images")
    print()

    # Generate priority list for each split
    print("📝 Generating priority annotation lists...")

    for split in ['train', 'valid', 'test']:
        split_invalid = invalid_images.get(split, [])

        # Save priority list
        priority_file = output_dir / f"{split}_priority_images.txt"
        with open(priority_file, 'w') as f:
            f.write(f"# PlantVillage {split.upper()} - Priority Images (Invalid BBoxes)\n")
            f.write(f"# Total: {len(split_invalid)} images\n")
            f.write(f"#\n")
            f.write(f"# Annotate these images FIRST to fix invalid bounding boxes\n")
            f.write(f"#\n\n")

            for i, img_name in enumerate(sorted(split_invalid), 1):
                # Try to find the actual image file
                images_dir = plantvillage_root / split / "images"
                img_path = None
                for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                    potential = images_dir / f"{img_name}{ext}"
                    if potential.exists():
                        img_path = potential
                        break

                if img_path:
                    # Write relative path from plantvillage_root
                    rel_path = img_path.relative_to(plantvillage_root)
                    f.write(f"{i}. {rel_path}\n")
                else:
                    f.write(f"{i}. {img_name} (NOT FOUND)\n")

        print(f"   ✅ {priority_file.name}")

    print()

    # Generate dataset statistics
    print("📊 Dataset Statistics:")
    print()

    stats = {}
    for split in ['train', 'valid', 'test']:
        split_path = plantvillage_root / split
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        if not images_dir.exists():
            continue

        # Count files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))

        # Count instances and classes
        healthy_count = 0
        leaf_scorch_count = 0
        total_instances = 0

        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            total_instances += 1
                            if class_id == 0:
                                leaf_scorch_count += 1
                            elif class_id == 1:
                                healthy_count += 1

        stats[split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'total_instances': total_instances,
            'healthy': healthy_count,
            'leaf_scorch': leaf_scorch_count,
            'invalid_bboxes': len(invalid_images.get(split, []))
        }

        print(f"   {split.upper()}:")
        print(f"      Images: {stats[split]['images']}")
        print(f"      Labels: {stats[split]['labels']}")
        print(f"      Instances: {stats[split]['total_instances']}")
        print(f"         Healthy: {stats[split]['healthy']} ({stats[split]['healthy']/stats[split]['total_instances']*100:.1f}%)")
        print(f"         Leaf Scorch: {stats[split]['leaf_scorch']} ({stats[split]['leaf_scorch']/stats[split]['total_instances']*100:.1f}%)")
        print(f"      Invalid BBoxes: {stats[split]['invalid_bboxes']}")
        print()

    # Save stats as JSON
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   💾 Saved statistics: {stats_file.name}")
    print()

    # Generate CVAT import instructions
    instructions_file = output_dir / "CVAT_IMPORT_INSTRUCTIONS.md"
    with open(instructions_file, 'w') as f:
        f.write("# CVAT Import Instructions for PlantVillage\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Images:** {sum(s['images'] for s in stats.values())}\n")
        f.write(f"- **Total Instances:** {sum(s['total_instances'] for s in stats.values())}\n")
        f.write(f"- **Healthy Instances:** {sum(s['healthy'] for s in stats.values())}\n")
        f.write(f"- **Invalid BBoxes:** {total_invalid}\n\n")

        f.write("## Step-by-Step Import Process\n\n")

        f.write("### 1. Start CVAT\n\n")
        f.write("```bash\n")
        f.write("./scripts/setup_cvat.sh\n")
        f.write("```\n\n")

        f.write("### 2. Create Superuser\n\n")
        f.write("```bash\n")
        f.write("docker exec -it cvat python3 manage.py createsuperuser\n")
        f.write("```\n\n")
        f.write("Follow prompts to create username and password.\n\n")

        f.write("### 3. Access CVAT\n\n")
        f.write("Open browser: http://localhost:8080\n")
        f.write("Login with your credentials.\n\n")

        f.write("### 4. Create Project\n\n")
        f.write("1. Click **Projects** → **Create New Project**\n")
        f.write("2. **Name:** `PlantVillage_Healthy_Reannotation`\n")
        f.write("3. **Labels:**\n")
        f.write("   - Add label: `Strawberry___Leaf_scorch` (class 0)\n")
        f.write("   - Add label: `Strawberry___healthy` (class 1)\n")
        f.write("4. Click **Submit**\n\n")

        f.write("### 5. Create Tasks (One per Split)\n\n")

        for split in ['train', 'valid', 'test']:
            if split not in stats:
                continue

            f.write(f"\n#### Task: {split.upper()} Split\n\n")
            f.write("1. Click **Tasks** → **Create New Task**\n")
            f.write(f"2. **Name:** `PlantVillage_{split.capitalize()}`\n")
            f.write(f"3. **Project:** Select `PlantVillage_Healthy_Reannotation`\n")
            f.write("4. **Select files:**\n")
            f.write("   - Method: **Share** (mounted directory)\n")
            f.write(f"   - Path: `/home/django/share/plantvillage/{split}/images/`\n")
            f.write("   - Select all images\n")
            f.write("5. **Advanced configuration:**\n")
            f.write("   - Image quality: 95\n")
            f.write("   - Overlap: 0\n")
            f.write("   - Segment size: 500\n")
            f.write("   - Check: **Use zip chunks**\n")
            f.write("6. Click **Submit**\n")
            f.write("7. Wait for task creation to complete\n")
            f.write("8. **Upload annotations** (optional but recommended):\n")
            f.write("   - Task → Actions → **Upload annotations**\n")
            f.write("   - Format: **YOLO 1.1**\n")
            f.write("   - Select labels from: `/home/django/share/plantvillage/{split}/labels/`\n")
            f.write("   - Click **OK**\n\n")
            f.write(f"   **Stats:** {stats[split]['images']} images, {stats[split]['healthy']} healthy instances, {stats[split]['invalid_bboxes']} to fix\n\n")

        f.write("### 6. Start Annotating\n\n")
        f.write("#### Priority 1: Fix Invalid BBoxes\n\n")
        f.write("Use the priority lists to focus on images that need fixing:\n\n")
        for split in ['train', 'valid', 'test']:
            if split in invalid_images and invalid_images[split]:
                f.write(f"- **{split.capitalize()}:** See `{split}_priority_images.txt` ({len(invalid_images[split])} images)\n")
        f.write("\n")

        f.write("#### Priority 2: Review All Healthy Samples\n\n")
        f.write("After fixing invalid bboxes, review all healthy instances for quality.\n\n")

        f.write("#### Priority 3: Full Dataset (Optional)\n\n")
        f.write("If time permits, review entire dataset for missed annotations.\n\n")

        f.write("### 7. Export Annotations\n\n")
        f.write("When annotation is complete:\n\n")
        f.write("1. Go to each Task\n")
        f.write("2. Actions → **Export annotations**\n")
        f.write("3. Format: **YOLO 1.1**\n")
        f.write("4. Download ZIP\n")
        f.write("5. Extract to: `data/external/plantvillage_healthy_reannotated/{split}/`\n\n")

        f.write("### 8. Validate\n\n")
        f.write("```bash\n")
        f.write("python scripts/validate_cvat_export.py \\\n")
        f.write("  --input data/external/plantvillage_healthy_reannotated/ \\\n")
        f.write("  --original data/external/plantvillage_strawberry/\n")
        f.write("```\n\n")

        f.write("## Annotation Guidelines\n\n")
        f.write("See detailed guidelines: `docs/CVAT_ANNOTATION_GUIDELINES.md`\n\n")

        f.write("## Troubleshooting\n\n")
        f.write("**CVAT not starting?**\n")
        f.write("```bash\n")
        f.write("docker compose -f docker-compose.cvat.yml down\n")
        f.write("docker compose -f docker-compose.cvat.yml up -d\n")
        f.write("```\n\n")

        f.write("**Can't see PlantVillage images?**\n")
        f.write("```bash\n")
        f.write("docker exec -it cvat ls -la /home/django/share/plantvillage/\n")
        f.write("```\n\n")

        f.write("**View CVAT logs:**\n")
        f.write("```bash\n")
        f.write("docker logs cvat -f\n")
        f.write("```\n\n")

    print(f"   📄 Generated instructions: {instructions_file.name}")
    print()

    print("=" * 70)
    print("✅ Preparation Complete!")
    print("=" * 70)
    print()
    print("📁 Output directory: cvat/import_data/")
    print()
    print("Generated files:")
    print(f"   1. train_priority_images.txt - {len(invalid_images.get('train', []))} images to fix")
    print(f"   2. valid_priority_images.txt - {len(invalid_images.get('valid', []))} images to fix")
    print(f"   3. test_priority_images.txt - {len(invalid_images.get('test', []))} images to fix")
    print("   4. dataset_statistics.json - Full dataset stats")
    print("   5. CVAT_IMPORT_INSTRUCTIONS.md - Step-by-step guide")
    print()
    print("📖 Next steps:")
    print("   1. Read: cvat/import_data/CVAT_IMPORT_INSTRUCTIONS.md")
    print("   2. Read: docs/CVAT_ANNOTATION_GUIDELINES.md")
    print("   3. Run: ./scripts/setup_cvat.sh")
    print("   4. Start annotating!")
    print()


if __name__ == "__main__":
    main()
