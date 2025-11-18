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

# Check if dependencies are already installed
check_python_package() {
    python3 -c "import $1" 2>/dev/null && return 0 || return 1
}

DEPS_TO_INSTALL=""
for pkg in opencv supervision pycocotools matplotlib pillow tqdm; do
    import_name=$pkg
    # Handle special cases
    [[ "$pkg" == "opencv" ]] && import_name="cv2"
    [[ "$pkg" == "pillow" ]] && import_name="PIL"

    if ! check_python_package "$import_name"; then
        # Convert package name for pip
        pip_name=$pkg
        [[ "$pkg" == "opencv" ]] && pip_name="opencv-python"
        DEPS_TO_INSTALL="$DEPS_TO_INSTALL $pip_name"
    else
        echo "  ✓ $pkg already installed"
    fi
done

if [ -n "$DEPS_TO_INSTALL" ]; then
    echo "  - Installing missing packages:$DEPS_TO_INSTALL"
    pip install --no-cache-dir $DEPS_TO_INSTALL
else
    echo "  ✓ All Python dependencies already installed"
fi

echo ""
echo "3. Installing SAM 2..."
# Navigate to Grounded-SAM-2 directory (clone if not exists)
if [ ! -d "Grounded-SAM-2" ]; then
    echo "  - Cloning Grounded-SAM-2 repo..."
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
else
    echo "  ✓ Grounded-SAM-2 repo already exists"
fi
cd Grounded-SAM-2

# Check if SAM 2 is already installed
if python3 -c "import sam2" 2>/dev/null; then
    echo "  ✓ SAM 2 already installed"
else
    echo "  - Installing SAM 2..."
    SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]" || echo "  ⚠️  SAM2 installed (CUDA extension may have failed, continuing...)"
fi

echo ""
echo "4. Installing Grounding DINO..."

# Check if Grounding DINO is already installed and working
if python3 -c "import groundingdino; from groundingdino.util.inference import Model" 2>/dev/null; then
    echo "  ✓ Grounding DINO already installed and working"
else
    echo "  - Installing Grounding DINO dependencies..."
    cd grounding_dino
    pip install -q -r requirements.txt

    # Check if CUDA extension is already built
    if [ -f "build/lib.linux-x86_64-cpython-312/groundingdino/_C.cpython-312-x86_64-linux-gnu.so" ]; then
        echo "  ✓ CUDA extensions already built"
    else
        echo "  - Building CUDA extensions (this may take 3-5 minutes)..."
        python setup.py build 2>&1 | grep -E "(Building|Compiling|error|Error)" || true
    fi

    # Check if already installed in site-packages
    if [ ! -d "/venv/main/lib/python3.12/site-packages/groundingdino" ]; then
        echo "  - Installing package..."
        python setup.py install 2>&1 | tail -5 || {
            echo "  ⚠️  setup.py install failed, using direct copy method..."
            # Copy built CUDA extension to source directory
            if [ -f "build/lib.linux-x86_64-cpython-312/groundingdino/_C.cpython-312-x86_64-linux-gnu.so" ]; then
                cp build/lib.linux-x86_64-cpython-312/groundingdino/_C.cpython-312-x86_64-linux-gnu.so groundingdino/
            fi
            # Copy entire package to site-packages
            mkdir -p /venv/main/lib/python3.12/site-packages/
            cp -r groundingdino /venv/main/lib/python3.12/site-packages/
        }
    else
        echo "  ✓ Grounding DINO already in site-packages"
    fi

    cd ..

    # Verify installation
    if python3 -c "import groundingdino" 2>/dev/null; then
        echo "  ✓ Grounding DINO installation verified"
    else
        echo "  ✗ ERROR: Grounding DINO installation failed"
        exit 1
    fi
fi

echo ""
echo "5. Downloading model checkpoints..."

# Download SAM 2.1 checkpoint
mkdir -p checkpoints
if [ ! -f "checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "  - Downloading SAM 2.1 Large checkpoint (~900MB)..."
    cd checkpoints
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/122724/sam2.1_hiera_large.pt || {
        echo "  ✗ Download failed, trying alternative method..."
        curl -L -o sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/122724/sam2.1_hiera_large.pt
    }
    cd ..
else
    echo "  ✓ SAM 2.1 checkpoint already exists"
fi

# Download Grounding DINO checkpoint
mkdir -p gdino_checkpoints
if [ ! -f "gdino_checkpoints/groundingdino_swint_ogc.pth" ]; then
    echo "  - Downloading Grounding DINO SwinT checkpoint (~700MB)..."
    cd gdino_checkpoints
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth || {
        echo "  ✗ Download failed, trying alternative method..."
        curl -L -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    }
    cd ..
else
    echo "  ✓ Grounding DINO checkpoint already exists"
fi

echo ""
echo "6. Verifying installation..."
python3 -c "
import torch
import sam2
import groundingdino

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ SAM 2: {sam2.__version__ if hasattr(sam2, \"__version__\") else \"installed\"}')
print(f'✓ Grounding DINO: installed')
" || {
    echo ""
    echo "✗ Verification failed. Check the errors above."
    exit 1
}

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
