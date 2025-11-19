"""Utilities for managing user prediction history."""

import os
import json
from typing import List, Dict, Any


BASE_DIR = os.path.join(os.getcwd(), "user_history")
os.makedirs(BASE_DIR, exist_ok=True)


def _user_file(username: str) -> str:
    """Get the file path for a user's history."""
    return os.path.join(BASE_DIR, f"{username}.json")


def load_history(username: str) -> List[Dict[str, Any]]:
    """
    Load prediction history for a user.
    
    Args:
        username: Username to load history for
        
    Returns:
        List of prediction records, empty list if no history exists
    """
    path = _user_file(username)
    if not os.path.exists(path):
        return []
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(username: str, record: Dict[str, Any]) -> None:
    """
    Save a prediction record to user's history.
    
    Args:
        username: Username to save history for
        record: Dictionary containing:
            - timestamp: ISO time string
            - inputs: dict of patient features
            - probability: float
            - prediction: int (0 or 1)
            - risk_level: str
            - guidance: dict
    """
    hist = load_history(username)
    hist.append(record)
    
    path = _user_file(username)
    with open(path, "w") as f:
        json.dump(hist, f, indent=2)


def last_n_history(username: str, n: int = 10) -> List[Dict[str, Any]]:
    """
    Get the last N prediction records for a user.
    
    Args:
        username: Username to get history for
        n: Number of recent records to return
        
    Returns:
        List of the last N prediction records
    """
    hist = load_history(username)
    return hist[-n:] if len(hist) > n else hist

