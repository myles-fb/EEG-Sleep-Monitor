"""Patient CRUD operations."""

from sqlalchemy.orm import joinedload

from models import Patient, get_db


def create_patient(name, age=None, study_type="single_night", **kwargs):
    with get_db() as db:
        patient = Patient(name=name, age=age, study_type=study_type, **kwargs)
        db.add(patient)
        db.flush()
        return patient


def get_patient(patient_id):
    with get_db() as db:
        return (
            db.query(Patient)
            .options(joinedload(Patient.studies))
            .filter_by(id=patient_id)
            .first()
        )


def list_patients():
    with get_db() as db:
        return (
            db.query(Patient)
            .options(joinedload(Patient.studies))
            .order_by(Patient.created_at.desc())
            .all()
        )


def update_patient(patient_id, **kwargs):
    with get_db() as db:
        patient = db.query(Patient).filter_by(id=patient_id).first()
        if patient:
            for k, v in kwargs.items():
                if hasattr(patient, k):
                    setattr(patient, k, v)
        return patient


def delete_patient(patient_id):
    with get_db() as db:
        patient = db.query(Patient).filter_by(id=patient_id).first()
        if patient:
            db.delete(patient)
            return True
        return False
