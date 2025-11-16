# Strawberry Disease Detection System

An automated disease detection and risk assessment system for soil-less strawberry greenhouses using YOLOv8 computer vision.

## Overview

This system monitors strawberries in a greenhouse using fixed cameras, detects diseases over time, generates risk maps by row/block, provides trend analysis, and sends Telegram alerts when disease risks are detected.

### Key Features

- **Real-time Disease Detection**: YOLOv8-based detection of 7 strawberry diseases
- **Risk Assessment**: Intelligent risk scoring based on disease type and confidence
- **Row-level Monitoring**: Track disease risk by greenhouse row/block
- **Trend Analysis**: Historical risk data with time-series optimization
- **Grafana Dashboards**: Visual heat maps, trend graphs, and statistics
- **Telegram Alerts**: Automated notifications when risk thresholds are exceeded
- **REST API**: Full API access to all system data
- **Docker Deployment**: Complete containerized stack

## Disease Classes

The system detects the following strawberry diseases:

1. **angular_leafspot** - Leaf spot caused by angular bacteria
2. **anthracnose_fruit_rot** - Severe fruit rot
3. **blossom_blight** - Blossom infection
4. **gray_mold** - Botrytis gray mold (high risk)
5. **leaf_spot** - Common leaf spot
6. **powdery_mildew_leaf** - Powdery mildew on leaves
7. **powdery_mildew_fruit** - Powdery mildew on fruit
8. **healthy** - No disease detected

## System Architecture

### Components

1. **Camera Collection Service**: Captures images from cameras (or simulates with sample images)
2. **Inference Service**: Runs YOLOv8 disease detection and calculates risk scores
3. **Risk Aggregation Service**: Aggregates risk data by row and time period
4. **FastAPI Service**: Provides REST API endpoints
5. **Telegram Notifier**: Sends alerts for high-risk conditions
6. **Grafana Dashboard**: Visualizes risk data
7. **PostgreSQL + TimescaleDB**: Time-series optimized database

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Kaggle API credentials (for dataset download)
- Python 3.11+ (for local development)

### Installation

1. **Clone the repository**:
   ```bash
   cd strawberry-disease
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Download and prepare dataset**:
   ```bash
   # Install Kaggle CLI
   pip install kaggle

   # Configure Kaggle credentials
   # Place your kaggle.json in ~/.kaggle/

   # Download dataset
   ./scripts/download_dataset.sh
   ```

4. **Train the YOLO model**:
   ```bash
   python scripts/train_model.py --epochs 100 --batch 16
   ```

   The trained model will be saved to `models/weights/best.pt`.

5. **Start the system**:
   ```bash
   docker-compose up -d
   ```

6. **Run database migrations**:
   ```bash
   docker-compose exec api alembic upgrade head
   ```

### Configuration

Key environment variables in `.env`:

- **Database**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- **API**: `API_TOKEN` (for authentication)
- **Telegram**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- **Risk Thresholds**: `HIGH_RISK_THRESHOLD`, `TELEGRAM_ALERT_THRESHOLD`
- **Intervals**: `CAMERA_CAPTURE_INTERVAL_MINUTES`, `RISK_AGGREGATION_INTERVAL_HOURS`

## Usage

### Accessing Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **PostgreSQL**: localhost:5432

### API Examples

**Get dashboard overview**:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/dashboard/overview
```

**Get high-risk rows**:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/rows/high-risk?threshold=70"
```

**Get row trend**:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/rows/row_1/trend?hours=24"
```

### Camera Configuration

Edit `config/cameras.yaml` to configure your cameras:

```yaml
cameras:
  - name: "camera_row_1"
    location: "Greenhouse North - Row 1"
    row_id: "row_1"
    stream_url: "rtsp://camera-ip:554/stream"  # Or null for simulation
    is_active: true
```

For simulation mode, place sample images in `data/raw/test/images/`.

## Development

### Project Structure

```
strawberry-disease/
├── src/
│   ├── api/                 # FastAPI application
│   ├── core/                # Core utilities, config, ML
│   │   └── ml/              # YOLO wrapper, risk calculator
│   ├── models/              # SQLAlchemy database models
│   └── services/            # Microservices
├── config/                  # Configuration files
│   ├── cameras.yaml         # Camera configuration
│   └── grafana/             # Grafana dashboards
├── scripts/                 # Utility scripts
├── docker/                  # Dockerfiles
├── alembic/                 # Database migrations
├── data/                    # Data storage
│   ├── raw/                 # Dataset
│   ├── images/              # Captured images
│   └── processed/           # Processed data
├── models/weights/          # YOLO model weights
├── tests/                   # Tests
└── docker-compose.yml       # Docker orchestration
```

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start individual services
python -m src.services.camera_collector
python -m src.services.inference_service
python -m src.services.risk_aggregator
python -m src.services.telegram_notifier
python -m src.api.main
```

### Running Tests

```bash
pytest tests/ -v --cov=src
```

## Risk Calculation

Risk scores (0-100) are calculated based on:

1. **Disease Type**: Each disease has a base risk range
   - High risk: gray_mold (60-90), anthracnose (65-95), blossom_blight (70-95)
   - Medium: powdery_mildew (40-70), leaf_spot (30-50)
   - Low: healthy (0-10)

2. **Model Confidence**: Higher confidence increases risk within the range

3. **Aggregation**:
   - Image risk = Maximum risk among all detections
   - Row risk = Average/max risk over time window

## Monitoring

### Grafana Dashboards

The system includes a pre-configured Grafana dashboard with:

- System overview statistics
- Risk heat map by row
- 24-hour risk trends
- Disease distribution
- Top high-risk rows
- Image processing status

### Telegram Alerts

Configure alerts in `.env`:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ALERT_THRESHOLD=70
TELEGRAM_NOTIFICATION_COOLDOWN_MINUTES=60
```

Alerts include:
- Row ID and risk level
- Average and maximum risk scores
- Dominant disease
- Visual risk indicator

## Database Schema

### Tables

- **cameras**: Camera metadata and configuration
- **images**: Captured images with processing status
- **predictions**: Individual disease detections
- **risk_summaries**: Aggregated risk data (TimescaleDB hypertable)

### Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Troubleshooting

### Model not found error
Ensure you've trained the model and placed `best.pt` in `models/weights/`.

### No sample images in simulation mode
Place test images in `data/raw/test/images/` or configure a real camera stream.

### Database connection errors
Check that PostgreSQL container is running: `docker-compose ps postgres`

### Telegram notifications not sending
Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are correctly configured.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

- **Dataset**: [Strawberry Disease Detection Dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Framework**: FastAPI, SQLAlchemy, Grafana