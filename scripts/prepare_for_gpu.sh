#!/bin/bash
# Prepare all files for GPU upload
# This script packages images and scripts into tarballs ready for vast.ai upload

set -e

echo "=========================================="
echo "Preparing files for GPU upload"
echo "=========================================="

PROJECT_DIR="/Users/errorist/Documents/personal-projects/strawberry-disease"
cd "$PROJECT_DIR"

echo ""
echo "1. Packaging images..."
if [ -d "data/external/kaggle_tipburn/train/images" ]; then
    # Create images tarball
    tar -czf kaggle_images.tar.gz \
        -C data/external/kaggle_tipburn/train \
        images/

    IMAGE_SIZE=$(du -h kaggle_images.tar.gz | cut -f1)
    IMAGE_COUNT=$(find data/external/kaggle_tipburn/train/images -type f | wc -l | tr -d ' ')

    echo "  ✓ Created kaggle_images.tar.gz ($IMAGE_SIZE, $IMAGE_COUNT images)"
else
    echo "  ✗ ERROR: Images directory not found!"
    echo "    Expected: data/external/kaggle_tipburn/train/images"
    exit 1
fi

echo ""
echo "2. Packaging scripts..."
tar -czf grounded_sam2_scripts.tar.gz \
    scripts/setup_grounded_sam2_gpu.sh \
    scripts/batch_process_kaggle.py \
    scripts/convert_json_to_yolo.py

SCRIPTS_SIZE=$(du -h grounded_sam2_scripts.tar.gz | cut -f1)
echo "  ✓ Created grounded_sam2_scripts.tar.gz ($SCRIPTS_SIZE)"

echo ""
echo "3. Verifying Grounded-SAM-2 repo..."
if [ -d "Grounded-SAM-2" ]; then
    REPO_SIZE=$(du -sh Grounded-SAM-2 | cut -f1)
    echo "  ✓ Grounded-SAM-2 repo exists ($REPO_SIZE)"
    echo "    Note: You'll upload this separately or clone on GPU server"
else
    echo "  ✗ WARNING: Grounded-SAM-2 repo not found!"
    echo "    You can clone it on the GPU server instead"
fi

echo ""
echo "4. Creating upload checklist..."
cat > gpu_upload_checklist.txt << 'EOF'
GPU Upload Checklist
====================

Files to upload to vast.ai:
---------------------------
[ ] kaggle_images.tar.gz        - 626 strawberry leaf images
[ ] grounded_sam2_scripts.tar.gz - Setup and processing scripts
[ ] Grounded-SAM-2/             - Model repo (or clone on server)

Before starting GPU:
--------------------
[ ] vast.ai account has credits
[ ] SSH key configured
[ ] Read docs/GPU_EXECUTION_GUIDE.md
[ ] Set a timer to remember to stop instance!

Upload commands:
----------------
# Replace XXXXX with your vast.ai port, XX.XXX.XX.XXX with IP

scp -P XXXXX kaggle_images.tar.gz root@XX.XXX.XX.XXX:/workspace/
scp -P XXXXX grounded_sam2_scripts.tar.gz root@XX.XXX.XX.XXX:/workspace/

# Option 1: Upload repo
scp -P XXXXX -r Grounded-SAM-2/ root@XX.XXX.XX.XXX:/workspace/

# Option 2: Clone on server (faster)
ssh -p XXXXX root@XX.XXX.XX.XXX
cd /workspace
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

After upload:
-------------
bash /workspace/scripts/setup_grounded_sam2_gpu.sh
python3 /workspace/scripts/batch_process_kaggle.py \
  --images_dir /workspace/train/images \
  --output_dir /workspace/grounded_sam2_results \
  --checkpoint_dir /workspace/Grounded-SAM-2

Download results:
-----------------
scp -P XXXXX -r root@XX.XXX.XX.XXX:/workspace/grounded_sam2_results/ ./data/external/

⚠️  STOP GPU INSTANCE when done!
EOF

echo "  ✓ Created gpu_upload_checklist.txt"

echo ""
echo "5. Generating summary..."
echo ""
echo "=========================================="
echo "✓ Preparation complete!"
echo "=========================================="
echo ""
echo "Files ready for upload:"
echo "  - kaggle_images.tar.gz ($IMAGE_SIZE)"
echo "  - grounded_sam2_scripts.tar.gz ($SCRIPTS_SIZE)"
echo "  - Grounded-SAM-2/ (clone on server or upload)"
echo ""
echo "Next steps:"
echo "  1. Read: docs/GPU_EXECUTION_GUIDE.md"
echo "  2. Review: gpu_upload_checklist.txt"
echo "  3. Start vast.ai GPU instance"
echo "  4. Upload files and run processing"
echo ""
echo "Estimated GPU time: 30-90 minutes"
echo "Estimated cost: \$0.25-\$0.65 (RTX 3090/4090)"
echo "=========================================="
