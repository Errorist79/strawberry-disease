#!/bin/bash

# Label Studio Setup Script for PlantVillage Re-annotation
# Much simpler alternative to CVAT

set -e

echo "======================================================================"
echo "Label Studio Setup for PlantVillage Re-annotation"
echo "======================================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed!"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not available!"
    exit 1
fi

echo "✅ Docker: $(docker --version)"
echo "✅ Docker Compose: $(docker compose version)"
echo ""

# Check if already running
if docker ps | grep -q labelstudio; then
    echo "⚠️  Label Studio is already running!"
    echo ""
    read -p "Restart? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping..."
        docker compose -f docker-compose.labelstudio.yml down
    else
        echo "Keeping existing container."
        exit 0
    fi
fi

echo "======================================================================"
echo "Starting Label Studio..."
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Pull Label Studio Docker image (~500MB)"
echo "  2. Start Label Studio server"
echo "  3. Mount PlantVillage dataset"
echo ""

# Start Label Studio
docker compose -f docker-compose.labelstudio.yml up -d

echo ""
echo "⏳ Waiting for Label Studio to start (15 seconds)..."
sleep 15

# Check status
echo ""
echo "📊 Container status:"
docker compose -f docker-compose.labelstudio.yml ps

echo ""
echo "======================================================================"
echo "✅ Label Studio is running!"
echo "======================================================================"
echo ""
echo "🌐 Web Interface: http://localhost:8080"
echo ""
echo "📋 Next Steps:"
echo "  1. Open browser: http://localhost:8080"
echo "  2. Sign up (create account - stored locally)"
echo "  3. Create project: PlantVillage_Healthy_Reannotation"
echo "  4. Configure labeling (Rectangle labels)"
echo "  5. Import PlantVillage images"
echo "  6. Start annotating!"
echo ""
echo "📁 Mounted directories:"
echo "   - PlantVillage: /label-studio/data/plantvillage/"
echo "   - Export folder: /label-studio/export/"
echo ""
echo "🛑 To stop:"
echo "   docker compose -f docker-compose.labelstudio.yml down"
echo ""
echo "🔄 To view logs:"
echo "   docker logs labelstudio -f"
echo ""
echo "======================================================================"
