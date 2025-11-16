"""Initial schema

Revision ID: 001_initial
Revises:
Create Date: 2025-01-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create cameras table
    op.create_table(
        'cameras',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('location', sa.String(length=200), nullable=False),
        sa.Column('row_id', sa.String(length=50), nullable=False),
        sa.Column('stream_url', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_cameras_name'), 'cameras', ['name'], unique=True)
    op.create_index(op.f('ix_cameras_row_id'), 'cameras', ['row_id'], unique=False)

    # Create images table
    op.create_table(
        'images',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('camera_id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('max_risk_score', sa.Integer(), nullable=True),
        sa.Column('dominant_disease', sa.String(length=50), nullable=True),
        sa.Column('prediction_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_path')
    )
    op.create_index(op.f('ix_images_camera_id'), 'images', ['camera_id'], unique=False)
    op.create_index(op.f('ix_images_status'), 'images', ['status'], unique=False)

    # Create predictions table
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('class_label', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('bbox_x', sa.Float(), nullable=True),
        sa.Column('bbox_y', sa.Float(), nullable=True),
        sa.Column('bbox_width', sa.Float(), nullable=True),
        sa.Column('bbox_height', sa.Float(), nullable=True),
        sa.Column('risk_score', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_image_id'), 'predictions', ['image_id'], unique=False)
    op.create_index(op.f('ix_predictions_class_label'), 'predictions', ['class_label'], unique=False)
    op.create_index(op.f('ix_predictions_risk_score'), 'predictions', ['risk_score'], unique=False)

    # Create risk_summaries table
    op.create_table(
        'risk_summaries',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('row_id', sa.String(length=50), nullable=False),
        sa.Column('time_bucket', sa.DateTime(timezone=True), nullable=False),
        sa.Column('avg_risk_score', sa.Float(), nullable=False),
        sa.Column('max_risk_score', sa.Integer(), nullable=False),
        sa.Column('min_risk_score', sa.Integer(), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('dominant_disease', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('row_id', 'time_bucket', name='uq_row_time_bucket')
    )
    op.create_index(op.f('ix_risk_summaries_row_id'), 'risk_summaries', ['row_id'], unique=False)
    op.create_index(op.f('ix_risk_summaries_time_bucket'), 'risk_summaries', ['time_bucket'], unique=False)

    # Enable TimescaleDB hypertable for risk_summaries (time-series optimization)
    op.execute("SELECT create_hypertable('risk_summaries', 'time_bucket', if_not_exists => TRUE);")


def downgrade() -> None:
    op.drop_index(op.f('ix_risk_summaries_time_bucket'), table_name='risk_summaries')
    op.drop_index(op.f('ix_risk_summaries_row_id'), table_name='risk_summaries')
    op.drop_table('risk_summaries')

    op.drop_index(op.f('ix_predictions_risk_score'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_class_label'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_image_id'), table_name='predictions')
    op.drop_table('predictions')

    op.drop_index(op.f('ix_images_status'), table_name='images')
    op.drop_index(op.f('ix_images_camera_id'), table_name='images')
    op.drop_table('images')

    op.drop_index(op.f('ix_cameras_row_id'), table_name='cameras')
    op.drop_index(op.f('ix_cameras_name'), table_name='cameras')
    op.drop_table('cameras')
