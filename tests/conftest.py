"""Cross-platform pytest runtime configuration shared by all RepoMind tests."""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

_CREATED_RUN_DIRECTORIES = pytest.StashKey[list[Path]]()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest artifacts project-local without sharing them across OS accounts.

    Every directory created here is unique to this run and recorded on the
    config so :func:`pytest_unconfigure` removes exactly those directories,
    never one supplied by the user (``--basetemp``/``cache_dir``) or created
    by a concurrent run.
    """
    component = uuid4().hex
    created: list[Path] = []
    config.stash[_CREATED_RUN_DIRECTORIES] = created
    if config.getoption("basetemp") is None:
        basetemp = Path(config.rootpath, f".pytest-tmp-{component}")
        config.option.basetemp = basetemp
        created.append(basetemp)
    # ``cache_dir`` is only registered by the cacheprovider plugin; with
    # ``-p no:cacheprovider`` there is no cache to relocate.
    if (
        config.pluginmanager.hasplugin("cacheprovider")
        and config.getini("cache_dir") == ".pytest_cache"
    ):
        cache_dir = f".pytest-cache-{component}"
        config.inicfg["cache_dir"] = cache_dir
        config._inicache["cache_dir"] = cache_dir
        created.append(Path(config.rootpath, cache_dir))


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the per-run directories this conftest created; they have no later use."""
    for directory in config.stash.get(_CREATED_RUN_DIRECTORIES, []):
        shutil.rmtree(directory, ignore_errors=True)
