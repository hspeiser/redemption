"""TOML configuration loading.

All runtime behaviour is driven by the ``configs/*.toml`` files -- there are no
command-line arguments anywhere in this project. This module loads those files
into lightweight attribute-accessible dictionaries (:class:`DotDict`) and
locates the ``configs`` directory relative to the repository root.

Typical use::

    from redemption.config import load_all
    cfg = load_all()            # cfg.camera, cfg.gate, cfg.datagen, ...
    fx = cfg.camera.intrinsics.fx
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class DotDict(dict):
    """A ``dict`` whose keys are also accessible as attributes (recursively).

    Read-only in spirit -- we only ever load config, never mutate it. Nested
    dicts and lists-of-dicts are converted on access-time construction.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [DotDict._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - surfaced as AttributeError
            raise AttributeError(
                f"config key {name!r} not found (available: {list(self.keys())})"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by ``"a.b.c"`` path, returning ``default`` if absent."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def repo_root() -> Path:
    """Return the repository root (two levels up from this file: src/redemption/)."""
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def load_toml(path: str | Path) -> DotDict:
    """Load a single TOML file into a :class:`DotDict`."""
    path = Path(path)
    if not path.is_absolute():
        path = configs_dir() / path
    with open(path, "rb") as fh:
        return DotDict(tomllib.load(fh))


def load_all() -> DotDict:
    """Load every project config into a single namespace.

    Returns a :class:`DotDict` with keys ``camera``, ``gate``, ``datagen``,
    ``train``, ``pnp`` and ``report``.
    """
    names = ["camera", "gate", "datagen", "train", "pnp", "report"]
    return DotDict({name: load_toml(f"{name}.toml") for name in names})
