"""DocChat: a from-scratch RAG library for question answering over documents."""

from .config import Settings, get_settings
from .pipeline import DocChat

__all__ = ["DocChat", "Settings", "get_settings"]
