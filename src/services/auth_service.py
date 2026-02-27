"""Authentication service — user registration and login."""

import bcrypt

from models import User, get_db


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(email: str, password: str, display_name: str) -> User:
    """Create a new user. Raises ValueError if email already exists."""
    with get_db() as db:
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            raise ValueError(f"Email '{email}' is already registered.")
        user = User(
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
        )
        db.add(user)
        db.flush()
        return user


def authenticate(email: str, password: str):
    """Return User if credentials valid, else None."""
    with get_db() as db:
        user = db.query(User).filter_by(email=email).first()
        if user and verify_password(password, user.password_hash):
            return user
        return None


def get_user(user_id: str):
    """Look up user by ID."""
    with get_db() as db:
        return db.query(User).filter_by(id=user_id).first()
