"""Fixtures for opt-in PostgreSQL + pgvector integration tests."""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from repomind.db import create_database_engine


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    database_url = os.environ.get("REPOMIND_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("set REPOMIND_TEST_DATABASE_URL to run real PostgreSQL tests")

    engine = create_database_engine(database_url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        pytest.fail(f"configured PostgreSQL test database is unavailable: {exc}")

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(postgres_engine: Engine) -> Iterator[Session]:
    connection = postgres_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
