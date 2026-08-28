"""Configuration loading.

Single source of truth for parameters. Everything in src/ reads from here
rather than hardcoding values, so experiments are reproducible by diffing
config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    """Thin wrapper over the YAML config with resolved absolute paths."""

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a key under `paths:` to an absolute path."""
        return (self.root / self.raw["paths"][key]).resolve()

    def raw_file(self, name: str) -> Path:
        """Absolute path to a file inside the raw data directory."""
        return self.path("raw") / name

    def ensure_dirs(self) -> None:
        for key in ("raw", "interim", "processed", "reports"):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=path.resolve().parent)
