from __future__ import annotations

import os


def configure_local_runtime_defaults() -> dict[str, str]:
    """Set safe process/thread defaults when not already configured."""
    defaults = {
        "OMP_NUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KMP_INIT_AT_FORK": "FALSE",
        "KMP_BLOCKTIME": "0",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    return defaults
