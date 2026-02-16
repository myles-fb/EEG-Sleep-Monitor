"""SQLAlchemy ORM models for the physician dashboard."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
)
from sqlalchemy.orm import relationship

from .database import Base


def _uuid():
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    study_type = Column(String, default="single_night")  # single_night | multi_night

    # Feature selection
    enable_mo_detection = Column(Boolean, default=True)
    enable_envelope_spectrogram = Column(Boolean, default=True)
    enable_full_spectrogram = Column(Boolean, default=True)

    # Time settings
    bucket_size_seconds = Column(Integer, default=3600)   # dashboard bucket (1 hr)
    window_size_seconds = Column(Integer, default=300)    # algorithm window (5 min)

    # Notification thresholds
    mo_count_threshold = Column(Integer, nullable=True)

    # Storage
    save_raw_eeg = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    studies = relationship(
        "Study", back_populates="patient", cascade="all, delete-orphan"
    )


class Device(Base):
    __tablename__ = "devices"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    device_key = Column(String, unique=True, nullable=False)  # hardware ID
    patient_id = Column(String, ForeignKey("patients.id"), nullable=True)
    current_study_id = Column(String, nullable=True)
    serial_port = Column(String, nullable=True)
    status = Column(String, default="offline")  # offline | connected | streaming | error
    last_heartbeat = Column(DateTime, nullable=True)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", backref="devices")


class Study(Base):
    __tablename__ = "studies"

    id = Column(String, primary_key=True, default=_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    device_id = Column(String, ForeignKey("devices.id"), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, default="active")   # active | completed
    source = Column(String, default="edf")      # edf | live | pi
    source_file = Column(String, nullable=True)
    duration_sec = Column(Float, nullable=True)
    sample_rate = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    channels_json = Column(Text, nullable=True)      # JSON: [{"index": 0, "label": "Fp1-F7"}, ...]

    patient = relationship("Patient", back_populates="studies")
    device = relationship("Device")
    feature_records = relationship(
        "FeatureRecord", back_populates="study", cascade="all, delete-orphan"
    )
    alerts = relationship(
        "Alert", back_populates="study", cascade="all, delete-orphan"
    )


class FeatureRecord(Base):
    """Time-series feature storage: one row per feature-key per algorithm window."""
    __tablename__ = "feature_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String, ForeignKey("studies.id"), nullable=False)
    timestamp = Column(Float, nullable=False)       # seconds from recording start
    bucket_start = Column(Float, nullable=False)
    bucket_end = Column(Float, nullable=False)
    feature_key = Column(String, nullable=False)     # e.g. mo_q_0.5_3hz, mo_count
    feature_value = Column(Float, nullable=True)     # null if feature disabled
    metadata_json = Column(Text, nullable=True)      # optional per-window detail
    channel_index = Column(Integer, nullable=True)   # bipolar channel index (NULL = legacy single-channel)
    channel_label = Column(String, nullable=True)    # e.g. "Fp1-F7"

    study = relationship("Study", back_populates="feature_records")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String, ForeignKey("studies.id"), nullable=False)
    timestamp = Column(Float, nullable=False)
    alert_type = Column(String, nullable=False)      # e.g. mo_count
    threshold = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=False)
    bucket_time = Column(String, nullable=True)      # e.g. "02:00-03:00"
    acknowledged = Column(Boolean, default=False)

    study = relationship("Study", back_populates="alerts")
