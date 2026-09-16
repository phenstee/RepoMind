"""PostgreSQL advisory locks for index/coding execution across processes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from repomind.api.errors import APIError


class RepositoryExecutionLock(Protocol):
    @contextmanager
    def hold(self, repository_id: int) -> Iterator[None]: ...


class NullRepositoryExecutionLock:
    @contextmanager
    def hold(self, repository_id: int) -> Iterator[None]:
        del repository_id
        yield


class PostgresRepositoryExecutionLock:
    """Session-scoped advisory lock shared by HTTP and worker index/coding execution."""

    _NAMESPACE = 1_902_019

    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    @contextmanager
    def hold(self, repository_id: int) -> Iterator[None]:
        with self.factory() as session:
            locked = session.scalar(
                text("SELECT pg_try_advisory_lock(:namespace, :repository_id)"),
                {"namespace": self._NAMESPACE, "repository_id": repository_id},
            )
            if locked is not True:
                raise APIError(409, "repository_busy", "Another repository operation is running.")
            try:
                yield
            finally:
                session.execute(
                    text("SELECT pg_advisory_unlock(:namespace, :repository_id)"),
                    {"namespace": self._NAMESPACE, "repository_id": repository_id},
                )
