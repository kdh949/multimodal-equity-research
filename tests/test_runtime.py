from __future__ import annotations

import os

from quant_research.runtime import configure_local_runtime_defaults


def test_configure_local_runtime_defaults_preserves_existing_values(monkeypatch: object) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    for key in [
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "KMP_BLOCKTIME",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY",
    ]:
        monkeypatch.delenv(key, raising=False)

    defaults = configure_local_runtime_defaults()

    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert defaults["KMP_INIT_AT_FORK"] == "FALSE"
    assert os.environ["KMP_INIT_AT_FORK"] == "FALSE"
    assert os.environ["KMP_BLOCKTIME"] == "0"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == "YES"


def test_configure_local_runtime_defaults_keeps_existing_non_default(monkeypatch: object) -> None:
    monkeypatch.setenv("KMP_BLOCKTIME", "17")
    defaults = configure_local_runtime_defaults()
    assert os.environ["KMP_BLOCKTIME"] == "17"
    assert os.environ["OMP_NUM_THREADS"] == defaults["OMP_NUM_THREADS"]
