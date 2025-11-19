"""User prediction history module."""

from .history_utils import load_history, save_history, last_n_history
from .ui import history_dashboard

__all__ = ["load_history", "save_history", "last_n_history", "history_dashboard"]

