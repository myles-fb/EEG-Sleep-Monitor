"""Patient CRUD operations."""

from sqlalchemy.orm import joinedload

from models import Patient, get_db


def create_patient(name, user_id=None, age=None, study_type="single_night", **kwargs):
    with get_db() as db:
        patient = Patient(
            name=name, age=age, study_type=study_type, user_id=user_id, **kwargs
        )
        db.add(patient)
        db.flush()
        return patient


def get_patient(patient_id, user_id=None):
    with get_db() as db:
        q = (
            db.query(Patient)
            .options(joinedload(Patient.studies))
            .filter_by(id=patient_id)
        )
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        return q.first()


def list_patients(user_id=None):
    with get_db() as db:
        q = (
            db.query(Patient)
            .options(joinedload(Patient.studies))
        )
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        return q.order_by(Patient.created_at.desc()).all()


def update_patient(patient_id, user_id=None, **kwargs):
    with get_db() as db:
        q = db.query(Patient).filter_by(id=patient_id)
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        patient = q.first()
        if patient:
            for k, v in kwargs.items():
                if hasattr(patient, k):
                    setattr(patient, k, v)
        return patient


def delete_patient(patient_id, user_id=None):
    with get_db() as db:
        q = db.query(Patient).filter_by(id=patient_id)
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        patient = q.first()
        if patient:
            db.delete(patient)
            return True
        return False
