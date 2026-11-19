"""
SQLAlchemy ORM models for crowd monitoring: zones, detection records, incidents, and notifications.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from src.storage.database import Base


class MonitoringZoneModel(Base):
    __tablename__ = "monitoring_zones"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    x_min_pct = Column(Float, nullable=False, default=0.1)
    y_min_pct = Column(Float, nullable=False, default=0.1)
    x_max_pct = Column(Float, nullable=False, default=0.9)
    y_max_pct = Column(Float, nullable=False, default=0.9)
    low_threshold = Column(Integer, nullable=False, default=2)
    moderate_threshold = Column(Integer, nullable=False, default=5)
    high_threshold = Column(Integer, nullable=False, default=8)
    critical_threshold = Column(Integer, nullable=False, default=10)
    sustained_seconds = Column(Float, nullable=False, default=3.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DetectionRecordModel(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    zone_id = Column(String(64), nullable=True, index=True)
    people_count = Column(Integer, nullable=False)
    density_status = Column(String(32), nullable=False)
    fps = Column(Float, nullable=True)


class IncidentModel(Base):
    __tablename__ = "incidents"

    id = Column(String(64), primary_key=True, index=True)
    zone_id = Column(String(64), nullable=False, index=True)
    zone_name = Column(String(128), nullable=False)
    location = Column(String(128), nullable=False, default="Surveillance Hub")
    people_count = Column(Integer, nullable=False)
    density_status = Column(String(32), nullable=False, default="CRITICAL")
    threshold_critical = Column(Integer, nullable=False)
    sustained_seconds = Column(Float, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    snapshot_path = Column(String(256), nullable=True)
    email_status = Column(String(32), nullable=False, default="PENDING")
    email_error = Column(Text, nullable=True)


class NotificationLogModel(Base):
    __tablename__ = "notification_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String(64), ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    recipient = Column(String(128), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(32), nullable=False)  # SENT, FAILED, SKIPPED
    delivery_time_ms = Column(Float, nullable=False, default=0.0)
    error_message = Column(Text, nullable=True)

