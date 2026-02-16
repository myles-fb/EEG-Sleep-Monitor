"""Tests for study_service channel-aware query helpers."""

import json
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base
from models.models import Patient, Study, FeatureRecord

# Patch database to use in-memory SQLite for tests
import models.database as db_mod


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Create an in-memory SQLite database for each test."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, expire_on_commit=False)

    from contextlib import contextmanager

    @contextmanager
    def test_get_db():
        session = TestSession()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(db_mod, "get_db", test_get_db)
    monkeypatch.setattr("models.get_db", test_get_db)

    # Seed test data
    with test_get_db() as db:
        patient = Patient(id="p1", name="Test Patient")
        db.add(patient)
        db.flush()

        study = Study(id="s1", patient_id="p1", source="edf",
                      channels_json=json.dumps([
                          {"index": 0, "label": "Fp1-F7"},
                          {"index": 1, "label": "F7-T7"},
                      ]))
        db.add(study)
        db.flush()

        # Add feature records for two channels
        for ts in [0.0, 300.0]:
            for ch_idx, ch_label in [(0, "Fp1-F7"), (1, "F7-T7")]:
                db.add(FeatureRecord(
                    study_id="s1", timestamp=ts,
                    bucket_start=ts, bucket_end=ts + 300,
                    feature_key="mo_q_0.5_3hz",
                    feature_value=0.5 + ch_idx * 0.1,
                    channel_index=ch_idx, channel_label=ch_label,
                ))
                db.add(FeatureRecord(
                    study_id="s1", timestamp=ts,
                    bucket_start=ts, bucket_end=ts + 300,
                    feature_key="mo_count",
                    feature_value=float(ch_idx + 1),
                    channel_index=ch_idx, channel_label=ch_label,
                ))

    yield


def test_get_study_channels():
    from services.study_service import get_study_channels
    channels = get_study_channels("s1")
    assert len(channels) == 2
    assert channels[0]["index"] == 0
    assert channels[0]["label"] == "Fp1-F7"
    assert channels[1]["index"] == 1


def test_get_study_channels_empty():
    from services.study_service import get_study_channels
    channels = get_study_channels("nonexistent")
    assert channels == []


def test_get_feature_timeseries_by_channel():
    from services.study_service import get_feature_timeseries_by_channel
    # Filter to channel 0
    ts, vals = get_feature_timeseries_by_channel("s1", "mo_q_0.5_3hz", channel_index=0)
    assert len(ts) == 2
    assert all(v == 0.5 for v in vals)

    # Filter to channel 1
    ts, vals = get_feature_timeseries_by_channel("s1", "mo_q_0.5_3hz", channel_index=1)
    assert len(ts) == 2
    assert all(v == pytest.approx(0.6) for v in vals)


def test_get_feature_timeseries_by_channel_all():
    from services.study_service import get_feature_timeseries_by_channel
    # No channel filter returns all records
    ts, vals = get_feature_timeseries_by_channel("s1", "mo_q_0.5_3hz", channel_index=None)
    assert len(ts) == 4  # 2 timestamps x 2 channels


def test_update_study_channels():
    from services.study_service import update_study_channels, get_study_channels
    new_channels = [{"index": 5, "label": "F3-C3"}]
    update_study_channels("s1", new_channels)
    result = get_study_channels("s1")
    assert len(result) == 1
    assert result[0]["index"] == 5
    assert result[0]["label"] == "F3-C3"
