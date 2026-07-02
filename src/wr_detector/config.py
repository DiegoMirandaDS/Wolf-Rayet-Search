"""Configuration loading and project-root path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def load_reference_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = load_yaml(config_path)
    paths_config = load_yaml(config["paths_config"])
    catalogs_config = load_yaml(config["catalogs_config"])
    filters_config = load_yaml(config["filters_config"])

    config["paths"] = paths_config
    config["catalogs"] = catalogs_config
    config["filters"] = filters_config
    return config


def load_simbad_negative_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = load_yaml(config_path)
    paths_config = load_yaml(config["paths_config"])
    catalogs_config = load_yaml(config["catalogs_config"])
    filters_config = load_yaml(config["filters_config"])

    config["paths"] = paths_config
    config["catalogs"] = catalogs_config
    config["filters"] = filters_config
    return config
