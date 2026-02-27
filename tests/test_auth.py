"""Tests for authentication service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
import pytest

# Use in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite://"

from models import init_db
from services import auth_service


@pytest.fixture(autouse=True)
def setup_db():
    """Create fresh tables for each test."""
    from models.database import Base, engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


def test_create_user():
    user = auth_service.create_user("test@example.com", "password123", "Test User")
    assert user.email == "test@example.com"
    assert user.display_name == "Test User"
    assert user.id is not None
    assert user.password_hash != "password123"


def test_duplicate_email_rejected():
    auth_service.create_user("dup@example.com", "pass1", "User One")
    with pytest.raises(ValueError, match="already registered"):
        auth_service.create_user("dup@example.com", "pass2", "User Two")


def test_authenticate_success():
    auth_service.create_user("login@example.com", "secret", "Login User")
    user = auth_service.authenticate("login@example.com", "secret")
    assert user is not None
    assert user.email == "login@example.com"


def test_authenticate_wrong_password():
    auth_service.create_user("wrong@example.com", "correct", "Wrong PW")
    user = auth_service.authenticate("wrong@example.com", "incorrect")
    assert user is None


def test_authenticate_nonexistent_email():
    user = auth_service.authenticate("nobody@example.com", "anything")
    assert user is None


def test_get_user():
    created = auth_service.create_user("get@example.com", "pass", "Get User")
    fetched = auth_service.get_user(created.id)
    assert fetched is not None
    assert fetched.email == "get@example.com"


def test_get_user_nonexistent():
    fetched = auth_service.get_user("nonexistent-id")
    assert fetched is None


def test_password_hashing():
    pw = "my_secure_password"
    hashed = auth_service.hash_password(pw)
    assert hashed != pw
    assert auth_service.verify_password(pw, hashed) is True
    assert auth_service.verify_password("wrong", hashed) is False
