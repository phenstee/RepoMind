"""Tests for lazy database setup and caller-visible transaction handling."""

from unittest.mock import MagicMock

import repomind.db.session as session_module
from repomind.config import Settings


def test_default_database_url_matches_local_compose_service() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url == (
        "postgresql+psycopg://repomind:repomind@localhost:5432/repomind"
    )


def test_database_module_has_no_global_engine_or_session() -> None:
    assert not hasattr(session_module, "engine")
    assert not hasattr(session_module, "session")


def test_engine_factory_is_lazy_and_hides_sql_logging(monkeypatch) -> None:
    expected_engine = MagicMock()
    create_engine = MagicMock(return_value=expected_engine)
    monkeypatch.setattr(session_module, "create_engine", create_engine)

    engine = session_module.create_database_engine(
        "postgresql+psycopg://user:secret@localhost/database"
    )

    assert engine is expected_engine
    create_engine.assert_called_once_with(
        "postgresql+psycopg://user:secret@localhost/database",
        pool_pre_ping=True,
    )


def test_session_scope_commits_successful_unit_of_work() -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)

    with session_module.session_scope(factory) as yielded:
        assert yielded is session

    session.commit.assert_called_once_with()
    session.rollback.assert_not_called()
    session.close.assert_called_once_with()


def test_session_scope_rolls_back_failed_unit_of_work() -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)

    try:
        with session_module.session_scope(factory):
            raise RuntimeError("forced failure")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    session.commit.assert_not_called()
    session.rollback.assert_called_once_with()
    session.close.assert_called_once_with()
