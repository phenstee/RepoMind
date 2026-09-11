"""Shared SQLAlchemy declarative base without connection side effects."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for RepoMind persistence-only ORM models."""
