#!/bin/bash
#
# Download strawberry disease detection dataset from Kaggle
#
# Prerequisites:
#   - Kaggle API installed: pip install kaggle
#   - Kaggle API credentials configured: ~/.kaggle/kaggle.json
#

set -e

# Configuration
DATASET_NAME="usmanafzaal/strawberry-disease-detection-dataset"
OUTPUT_DIR="data/raw"
ZIP_FILE="strawberry-disease-detection-dataset.zip"

echo "=== Strawberry Disease Dataset Downloader ==="
echo ""

# Check if kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Install it with: pip install kaggle"
    exit 1
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API credentials not found at ~/.kaggle/kaggle.json"
    echo ""
    echo "To set up Kaggle API:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create an API token (downloads kaggle.json)"
    echo "3. Move it to ~/.kaggle/kaggle.json"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Downloading dataset: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Download dataset
kaggle datasets download -d "$DATASET_NAME" -p "$OUTPUT_DIR"

# Unzip dataset
echo ""
echo "Extracting dataset..."
cd "$OUTPUT_DIR"
unzip -q "$ZIP_FILE"
rm "$ZIP_FILE"

echo ""
echo "Dataset downloaded and extracted successfully!"
echo ""
echo "Dataset location: $OUTPUT_DIR"
echo ""
ls -lh

exit 0
