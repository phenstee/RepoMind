"""Offline command coverage for the versioned Retrieval V2 fixture."""

from pathlib import Path
from runpy import run_path


def test_repo_eval_v2_command_reports_strategy_matrix(capsys) -> None:
    benchmark = run_path(
        Path(__file__).parents[3] / "benchmarks" / "repo_eval_v2.py",
        run_name="repo_eval_v2_test",
    )
    benchmark["main"]()

    output = capsys.readouterr().out
    assert "REPO-EVAL-V2" in output
    assert "line_v1+exact" in output
    assert "python_ast_v1+exact" in output
    assert "python_ast_v1+exact+bm25_rrf" in output
    assert "Whole-symbol containment: 4/4" in output
    assert "not evidence about OpenAI embedding quality" in output
