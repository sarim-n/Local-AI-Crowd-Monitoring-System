"""
Repository functions for CRUD operations on zones, incidents, detection logs, and notifications.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from src.storage.models import (
    DetectionRecordModel,
    IncidentModel,
    MonitoringZoneModel,
    NotificationLogModel,
)


class MonitoringRepository:
    """Repository handling database operations."""

    def __init__(self, db: Session):
        self.db = db

    # ----------------- Zones -----------------
    def get_zones(self) -> List[MonitoringZoneModel]:
        return self.db.query(MonitoringZoneModel).all()

    def get_zone(self, zone_id: str) -> Optional[MonitoringZoneModel]:
        return self.db.query(MonitoringZoneModel).filter(MonitoringZoneModel.id == zone_id).first()

    def upsert_zone(
        self,
        zone_id: str,
        name: str,
        x_min_pct: float,
        y_min_pct: float,
        x_max_pct: float,
        y_max_pct: float,
        low_threshold: int = 2,
        moderate_threshold: int = 5,
        high_threshold: int = 8,
        critical_threshold: int = 10,
        sustained_seconds: float = 3.0,
    ) -> MonitoringZoneModel:
        zone = self.get_zone(zone_id)
        if zone:
            zone.name = name
            zone.x_min_pct = x_min_pct
            zone.y_min_pct = y_min_pct
            zone.x_max_pct = x_max_pct
            zone.y_max_pct = y_max_pct
            zone.low_threshold = low_threshold
            zone.moderate_threshold = moderate_threshold
            zone.high_threshold = high_threshold
            zone.critical_threshold = critical_threshold
            zone.sustained_seconds = sustained_seconds
            zone.updated_at = datetime.utcnow()
        else:
            zone = MonitoringZoneModel(
                id=zone_id,
                name=name,
                x_min_pct=x_min_pct,
                y_min_pct=y_min_pct,
                x_max_pct=x_max_pct,
                y_max_pct=y_max_pct,
                low_threshold=low_threshold,
                moderate_threshold=moderate_threshold,
                high_threshold=high_threshold,
                critical_threshold=critical_threshold,
                sustained_seconds=sustained_seconds,
            )
            self.db.add(zone)
        self.db.commit()
        self.db.refresh(zone)
        return zone

    def delete_zone(self, zone_id: str) -> bool:
        zone = self.get_zone(zone_id)
        if zone:
            self.db.delete(zone)
            self.db.commit()
            return True
        return False

    # ----------------- Detection Records -----------------
    def log_detection(
        self,
        people_count: int,
        density_status: str,
        zone_id: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> DetectionRecordModel:
        record = DetectionRecordModel(
            zone_id=zone_id,
            people_count=people_count,
            density_status=density_status,
            fps=fps,
            timestamp=datetime.utcnow(),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    # ----------------- Incidents -----------------
    def create_incident(
        self,
        incident_id: str,
        zone_id: str,
        zone_name: str,
        location: str,
        people_count: int,
        density_status: str,
        threshold_critical: int,
        sustained_seconds: float,
        snapshot_path: Optional[str] = None,
        email_status: str = "PENDING",
        email_error: Optional[str] = None,
    ) -> IncidentModel:
        incident = IncidentModel(
            id=incident_id,
            zone_id=zone_id,
            zone_name=zone_name,
            location=location,
            people_count=people_count,
            density_status=density_status,
            threshold_critical=threshold_critical,
            sustained_seconds=sustained_seconds,
            snapshot_path=snapshot_path,
            email_status=email_status,
            email_error=email_error,
            started_at=datetime.utcnow(),
        )
        self.db.add(incident)
        self.db.commit()
        self.db.refresh(incident)
        return incident

    def get_incidents(self, limit: int = 50) -> List[IncidentModel]:
        return (
            self.db.query(IncidentModel)
            .order_by(IncidentModel.started_at.desc())
            .limit(limit)
            .all()
        )

    def get_incident(self, incident_id: str) -> Optional[IncidentModel]:
        return self.db.query(IncidentModel).filter(IncidentModel.id == incident_id).first()

    def update_incident_email_status(
        self, incident_id: str, email_status: str, email_error: Optional[str] = None
    ) -> Optional[IncidentModel]:
        incident = self.get_incident(incident_id)
        if incident:
            incident.email_status = email_status
            incident.email_error = email_error
            self.db.commit()
            self.db.refresh(incident)
        return incident

    def resolve_incident(self, incident_id: str) -> Optional[IncidentModel]:
        incident = self.get_incident(incident_id)
        if incident:
            incident.resolved_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(incident)
        return incident

    # ----------------- Notification Logs -----------------
    def log_notification(
        self,
        incident_id: str,
        recipient: str,
        status: str,
        delivery_time_ms: float = 0.0,
        error_message: Optional[str] = None,
    ) -> NotificationLogModel:
        log_entry = NotificationLogModel(
            incident_id=incident_id,
            recipient=recipient,
            status=status,
            delivery_time_ms=delivery_time_ms,
            error_message=error_message,
            timestamp=datetime.utcnow(),
        )
        self.db.add(log_entry)
        self.db.commit()
        self.db.refresh(log_entry)
        return log_entry

    def get_notifications(self, limit: int = 50) -> List[NotificationLogModel]:
        return (
            self.db.query(NotificationLogModel)
            .order_by(NotificationLogModel.timestamp.desc())
            .limit(limit)
            .all()
        )

