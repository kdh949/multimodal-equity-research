from __future__ import annotations

from quant_research.runtime import configure_local_runtime_defaults


def pytest_configure(config: object) -> None:
    del config
    configure_local_runtime_defaults()
