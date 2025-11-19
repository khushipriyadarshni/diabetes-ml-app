"""Authentication utilities for Streamlit app."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional


USERS_FILE = Path("users.json")


def load_users() -> Dict[str, Dict[str, str]]:
    """Load users from the local JSON file."""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_users(users: Dict[str, Dict[str, str]]) -> None:
    """Save users to the local JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def hash_password(password: str) -> str:
    """Return a SHA-256 hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password: str, stored_hash: str) -> bool:
    """Check if a password matches the stored hash."""
    return hash_password(password) == stored_hash


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    """Authenticate user using stored credentials."""
    users = load_users()
    user_data = users.get(username)
    if not user_data:
        return False, None
    if check_password(password, user_data.get("password_hash", "")):
        return True, {"username": username, "email": user_data.get("email", "")}
    return False, None

