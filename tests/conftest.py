"""Cross-platform pytest runtime configuration shared by all RepoMind tests."""

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest artifacts project-local without sharing them across OS accounts."""
    component = uuid4().hex
    if config.getoption("basetemp") is None:
        config.option.basetemp = Path(config.rootpath, f".pytest-tmp-{component}")
    if config.getini("cache_dir") == ".pytest_cache":
        cache_dir = f".pytest-cache-{component}"
        config.inicfg["cache_dir"] = cache_dir
        config._inicache["cache_dir"] = cache_dir
