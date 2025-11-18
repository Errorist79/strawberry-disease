#!/bin/bash

# CVAT Setup Script for PlantVillage Re-annotation
# This script sets up CVAT locally using Docker Compose

set -e

echo "======================================================================"
echo "CVAT Setup for PlantVillage Re-annotation"
echo "======================================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available!"
    echo "Please install Docker Compose v2"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo "✅ Docker Compose found: $(docker compose version)"
echo ""

# Check if CVAT is already running
if docker ps | grep -q cvat; then
    echo "⚠️  CVAT containers are already running!"
    echo ""
    read -p "Do you want to stop and restart? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing CVAT containers..."
        docker compose -f docker-compose.cvat.yml down
    else
        echo "Keeping existing containers running."
        exit 0
    fi
fi

echo "======================================================================"
echo "Starting CVAT Docker Compose..."
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Pull CVAT Docker images (~2-3GB)"
echo "  2. Create PostgreSQL database"
echo "  3. Start CVAT server, UI, and proxy"
echo "  4. Mount PlantVillage dataset to /home/django/share/plantvillage"
echo ""

# Start database and redis first
echo "Starting database and redis..."
docker compose -f docker-compose.cvat.yml up -d cvat_db cvat_redis

echo "⏳ Waiting for database to be ready (10 seconds)..."
sleep 10

# Initialize database
echo ""
echo "Initializing CVAT database..."
docker compose -f docker-compose.cvat.yml run --rm cvat init

# Start remaining services
echo ""
echo "Starting CVAT server, UI, and proxy..."
docker compose -f docker-compose.cvat.yml up -d

echo ""
echo "⏳ Waiting for services to be ready (20 seconds)..."
sleep 20

# Check if all containers are running
echo ""
echo "📊 Container status:"
docker compose -f docker-compose.cvat.yml ps

echo ""
echo "======================================================================"
echo "✅ CVAT is now running!"
echo "======================================================================"
echo ""
echo "🌐 Web Interface: http://localhost:8080"
echo ""
echo "📋 Next Steps:"
echo "  1. Open browser and go to http://localhost:8080"
echo "  2. Create superuser account (see below)"
echo "  3. Create project for PlantVillage re-annotation"
echo "  4. Import images from shared folder: /home/django/share/plantvillage/"
echo ""
echo "👤 Create superuser account:"
echo "   Run this command to create an admin user:"
echo "   docker exec -it cvat python3 manage.py createsuperuser"
echo ""
echo "📁 Mounted directories:"
echo "   - PlantVillage: /home/django/share/plantvillage/ (read-only)"
echo "   - Logs: /home/django/share/logs/ (read-only)"
echo ""
echo "🛑 To stop CVAT:"
echo "   docker compose -f docker-compose.cvat.yml down"
echo ""
echo "🔄 To view logs:"
echo "   docker compose -f docker-compose.cvat.yml logs -f"
echo ""
echo "======================================================================"
