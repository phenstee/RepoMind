"""Integrity checks for the repo-eval-v5 question sets."""

import subprocess
from pathlib import Path

import pytest

from benchmarks.repo_eval_v5_cases import CORPUS_COMMIT, HELD_OUT_SUITE, SUITE

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_dev_and_held_out_sets_do_not_overlap() -> None:
    dev_ids = {case.id for case in SUITE.cases}
    held_out_ids = {case.id for case in HELD_OUT_SUITE.cases}
    dev_queries = {case.query for case in SUITE.cases}
    held_out_queries = {case.query for case in HELD_OUT_SUITE.cases}

    assert len(SUITE.cases) == 44
    assert len(HELD_OUT_SUITE.cases) == 16
    assert not dev_ids & held_out_ids
    assert not dev_queries & held_out_queries


def _corpus_commit_available() -> bool:
    try:
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{CORPUS_COMMIT}^{{commit}}"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


@pytest.mark.skipif(
    not _corpus_commit_available(), reason="frozen corpus commit is not in this clone's history"
)
def test_every_gold_target_resolves_against_the_frozen_corpus() -> None:
    from benchmarks.repo_eval_v5 import load_corpus
    from repomind.evaluation.spans import resolve_suite

    snapshot = load_corpus()

    for suite in (SUITE, HELD_OUT_SUITE):
        gold = resolve_suite(suite, snapshot)
        assert set(gold) == {case.id for case in suite.cases}


@pytest.mark.skipif(
    not _corpus_commit_available(), reason="frozen corpus commit is not in this clone's history"
)
def test_corpus_ignores_crlf_checkout_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Git for Windows defaults to core.autocrlf=true; the live run on Windows
    # scored a CRLF corpus until load_corpus pinned LF.
    from benchmarks.repo_eval_v5 import load_corpus

    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")

    snapshot = load_corpus()

    assert snapshot.files
    assert not any("\r\n" in source.content for source in snapshot.files)
