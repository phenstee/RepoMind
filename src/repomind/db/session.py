"""Lazy database engine and transaction helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from repomind.config import get_settings


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a lazy SQLAlchemy engine without opening a connection."""

    return create_engine(
        database_url or get_settings().database_url,
        pool_pre_ping=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create an injectable SQLAlchemy session factory."""

    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit one caller-visible unit of work or roll it back on failure."""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
