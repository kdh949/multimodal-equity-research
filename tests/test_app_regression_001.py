from __future__ import annotations

import runpy


def test_app_spawn_child_import_does_not_render_streamlit_ui() -> None:
    # Regression: ISSUE-001 - Streamlit app crashed when model ablation spawned a child process.
    # Found by /qa on 2026-05-05.
    # Report: .gstack/qa-reports/qa-report-localhost-2026-05-05.md
    namespace = runpy.run_path("app.py", run_name="__mp_main__")

    assert callable(namespace["main"])
