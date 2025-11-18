#!/bin/bash
# Setup script for Grounded-SAM-2 on vast.ai GPU server
# This script prepares the environment for batch processing 626 Kaggle images
# Run this immediately after starting the GPU instance

set -e  # Exit on error

echo "=========================================="
echo "Grounded-SAM-2 Setup for vast.ai"
echo "=========================================="

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. CUDA may not be properly installed."
    echo "Checking nvidia-smi..."
    nvidia-smi || echo "ERROR: No NVIDIA GPU detected!"
fi

echo ""
echo "1. Updating system packages..."
# Minimal updates to save time
apt-get update -qq

echo ""
echo "2. Installing Python dependencies..."
# Install PyTorch with CUDA 12.1 support
#pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install --no-cache-dir \
    opencv-python \
    supervision \
    pycocotools \
    matplotlib \
    pillow \
    tqdm

echo ""
echo "3. Installing SAM 2..."
# Navigate to Grounded-SAM-2 directory (clone if not exists)
if [ ! -d "Grounded-SAM-2" ]; then
    echo "  - Cloning Grounded-SAM-2 repo..."
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
fi
cd Grounded-SAM-2

# Install SAM 2 (allow errors for CUDA extension, we can run without it)
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]" || echo "SAM2 installed (CUDA extension may have failed, continuing...)"

echo ""
echo "4. Installing Grounding DINO..."
# Install Grounding DINO (remove --no-build-isolation for venv compatibility)
pip install -e grounding_dino || {
    echo "⚠️  Grounding DINO build failed, trying alternative method..."
    cd grounding_dino
    pip install -r requirements.txt
    python setup.py build develop
    cd ..
}

echo ""
echo "5. Downloading model checkpoints..."
# Download SAM 2.1 checkpoint
mkdir -p checkpoints
if [ ! -f "checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "  - Downloading SAM 2.1 Large checkpoint (~900MB)..."
    cd checkpoints
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/122724/sam2.1_hiera_large.pt
    cd ..
else
    echo "  - SAM 2.1 checkpoint already exists"
fi

# Download Grounding DINO checkpoint
mkdir -p gdino_checkpoints
if [ ! -f "gdino_checkpoints/groundingdino_swint_ogc.pth" ]; then
    echo "  - Downloading Grounding DINO SwinT checkpoint (~700MB)..."
    cd gdino_checkpoints
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    cd ..
else
    echo "  - Grounding DINO checkpoint already exists"
fi

echo ""
echo "6. Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Extract images: tar -xzf kaggle_images.tar.gz"
echo "2. Run batch processing: python3 batch_process_kaggle.py"
echo "3. Download results when done"
echo ""
echo "Estimated processing time: 30-90 minutes for 626 images"
echo "=========================================="
