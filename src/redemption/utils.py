"""Small shared utilities: logging, seeding, timing, JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from rich.logging import RichHandler

_LOG_CONFIGURED = False


def get_logger(name: str = "redemption") -> logging.Logger:
    """Return a rich-formatted logger (configured once)."""
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def child_seed(master_seed: int, index: int) -> int:
    """Deterministically derive a per-item seed from a master seed + index.

    Uses numpy's SeedSequence so worker RNGs are independent yet reproducible.
    """
    ss = np.random.SeedSequence([int(master_seed), int(index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def rng_for(master_seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(child_seed(master_seed, index))


def resolve_workers(requested: int) -> int:
    """Translate a config ``workers`` value (0 = auto) into a concrete count."""
    if requested and requested > 0:
        return int(requested)
    return max(1, (os.cpu_count() or 2) - 1)


@contextmanager
def timer(logger: logging.Logger, label: str):
    start = time.perf_counter()
    logger.info(f"{label} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        logger.info(f"{label} done in {dt:.1f}s")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays."""

    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, cls=_NpEncoder)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def timestamp() -> str:
    """Filesystem-friendly timestamp, e.g. ``20260721-185900``."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
