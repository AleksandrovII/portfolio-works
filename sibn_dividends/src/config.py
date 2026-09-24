"""Loads config/assumptions.yaml. All model constants live there — nothing here."""

from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "assumptions.yaml"


class Config:
    """Thin dict wrapper with dotted-path access, e.g.
    cfg.get('netback.discount_usd_per_bbl.base')."""

    def __init__(self, data: dict[str, Any], version: str):
        self._data = data
        self.version = version

    def get(self, path: str, default: Any = ...) -> Any:
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                if default is not ...:
                    return default
                raise KeyError(f"Missing config key: {path!r}")
            node = node[key]
        return node

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def raw(self) -> dict[str, Any]:
        return self._data


@functools.lru_cache(maxsize=1)
def load_config(path: pathlib.Path | None = None) -> Config:
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, version=data["version"])
