from __future__ import annotations

from pathlib import Path


def test_optional_dependency_warning_baseline_is_documented() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    assert "## Optional Dependency Warning Baseline" in markdown
    assert "non-semantic baseline evidence" in markdown
    assert "uv --cache-dir .uv-cache run pytest -q" in markdown
    assert "`795 passed`" in markdown
    assert "No warning summary was printed by the verified full-suite command" in markdown
    assert "Pandas compatibility fixes applied in `src/quant_research/features/sec.py`" in markdown
    assert "SEC numeric fill call sites" in markdown
    assert "SEC timestamp forward-fill call sites" in markdown
    assert "SEC per-ticker concatenation" in markdown
    assert "sklearn/pipeline.py:61" in markdown
    assert "tests/test_walk_forward.py::test_walk_forward_preprocessing_is_fit_on_fold_train_only" in markdown
    assert "signal labels" in markdown
    assert "validation gate status" in markdown
    assert "model predictions" in markdown
    assert "backtest returns" in markdown
    assert "report metrics" in markdown


def test_optional_dependency_warning_baseline_documents_intentional_scope() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    assert "Intentionally addressed in this hardening pass:" in markdown
    assert "Added this validation-suite inventory as auditable evidence." in markdown
    assert "Fixed pandas SEC feature call sites without changing generated values." in markdown
    assert "Did not suppress, filter, or silence pandas warnings" in markdown
    assert "Did not change SEC feature semantics" in markdown


def test_optional_dependency_warning_baseline_lists_expected_warning_counts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markdown = (repo_root / "docs" / "risk-validation.md").read_text()

    expected_rows = {
        "| `.venv/lib/python3.12/site-packages/sklearn/pipeline.py:61` | sklearn `FutureWarning` for methods on an unfitted `Pipeline` | 2 |",
        "| SEC numeric fill call sites | Convert merged object columns with `pd.to_numeric(...).fillna(0.0)` before rolling calculations. |",
        "| SEC timestamp forward-fill call sites | Normalize timestamp columns with `timestamp_utc(...).ffill()` before propagation. |",
        "| SEC per-ticker concatenation | Normalize SEC numeric and timestamp dtypes before concatenating ticker frames. |",
    }

    missing_rows = sorted(row for row in expected_rows if row not in markdown)

    assert not missing_rows, "\n".join(missing_rows)
