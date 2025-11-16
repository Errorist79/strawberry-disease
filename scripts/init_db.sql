-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for the application
CREATE SCHEMA IF NOT EXISTS public;

-- Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    location VARCHAR(200) NOT NULL,
    row_id VARCHAR(50) NOT NULL,
    stream_url VARCHAR(500),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    description VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cameras_name ON cameras(name);
CREATE INDEX IF NOT EXISTS idx_cameras_row_id ON cameras(row_id);

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    file_path VARCHAR(500) UNIQUE NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    max_risk_score INTEGER,
    dominant_disease VARCHAR(50),
    prediction_count INTEGER NOT NULL DEFAULT 0,
    error_message VARCHAR(1000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_images_camera_id ON images(camera_id);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_width FLOAT,
    bbox_height FLOAT,
    risk_score INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_image_id ON predictions(image_id);
CREATE INDEX IF NOT EXISTS idx_predictions_class_label ON predictions(class_label);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_score ON predictions(risk_score);

-- Risk summaries table
CREATE TABLE IF NOT EXISTS risk_summaries (
    id SERIAL PRIMARY KEY,
    row_id VARCHAR(50) NOT NULL,
    time_bucket TIMESTAMPTZ NOT NULL,
    avg_risk_score FLOAT NOT NULL,
    max_risk_score INTEGER NOT NULL,
    min_risk_score INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    dominant_disease VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_row_time_bucket UNIQUE (row_id, time_bucket)
);

CREATE INDEX IF NOT EXISTS idx_risk_summaries_row_id ON risk_summaries(row_id);
CREATE INDEX IF NOT EXISTS idx_risk_summaries_time_bucket ON risk_summaries(time_bucket);

-- Convert risk_summaries to TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable('risk_summaries', 'time_bucket', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
