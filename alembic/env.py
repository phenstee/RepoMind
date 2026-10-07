"""Alembic migration environment for RepoMind's PostgreSQL schema."""

from logging.config import fileConfig

from alembic import context
from repomind.config import get_settings
from repomind.db import models as database_models
from repomind.db.base import Base
from repomind.db.models import HNSW_INDEX_NAME
from repomind.db.session import create_database_engine

del database_models  # Import registers all tables on Base.metadata.

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(object_, name, type_, reflected, compare_to):
    """Hide schema objects that migrations own but ORM metadata cannot express.

    The HNSW index is created with raw SQL (a cast expression, an operator
    class, and a partial predicate) and is deliberately absent from
    ``Base.metadata``. Without this filter, autogenerate and ``alembic check``
    would propose dropping it.
    """

    del object_, reflected, compare_to
    return not (type_ == "index" and name == HNSW_INDEX_NAME)


def run_migrations_offline() -> None:
    """Run migrations without constructing a live database connection."""

    database_url = get_settings().database_url
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations through the configured lazy engine."""

    connectable = create_database_engine(get_settings().database_url)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
