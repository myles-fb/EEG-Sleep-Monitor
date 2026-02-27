"""Device CRUD operations for Raspberry Pi management."""

from datetime import datetime

from sqlalchemy.orm import joinedload

from models import Device, get_db


def register_device(name, device_key, user_id=None, serial_port=None, ip_address=None):
    with get_db() as db:
        device = Device(
            name=name,
            device_key=device_key,
            user_id=user_id,
            serial_port=serial_port,
            ip_address=ip_address,
        )
        db.add(device)
        db.flush()
        return device


def get_device(device_id, user_id=None):
    with get_db() as db:
        q = (
            db.query(Device)
            .options(joinedload(Device.patient))
            .filter_by(id=device_id)
        )
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        return q.first()


def get_device_by_key(device_key):
    with get_db() as db:
        return (
            db.query(Device)
            .options(joinedload(Device.patient))
            .filter_by(device_key=device_key)
            .first()
        )


def list_devices(user_id=None):
    with get_db() as db:
        q = (
            db.query(Device)
            .options(joinedload(Device.patient))
        )
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        return q.order_by(Device.created_at.desc()).all()


def assign_patient(device_id, patient_id, study_id=None):
    with get_db() as db:
        device = db.query(Device).filter_by(id=device_id).first()
        if device:
            device.patient_id = patient_id
            device.current_study_id = study_id
        return device


def update_status(device_id, status):
    with get_db() as db:
        device = db.query(Device).filter_by(id=device_id).first()
        if device:
            device.status = status
        return device


def update_heartbeat(device_key, ip_address=None):
    with get_db() as db:
        device = db.query(Device).filter_by(device_key=device_key).first()
        if device:
            device.last_heartbeat = datetime.utcnow()
            device.status = "connected"
            if ip_address:
                device.ip_address = ip_address
        return device


def delete_device(device_id, user_id=None):
    with get_db() as db:
        q = db.query(Device).filter_by(id=device_id)
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        device = q.first()
        if device:
            db.delete(device)
            return True
        return False
