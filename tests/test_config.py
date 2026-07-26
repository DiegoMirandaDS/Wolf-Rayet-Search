from __future__ import annotations

from pathlib import Path

import pytest

from wr_detector.config import load_yaml_with_extends


def test_yaml_extends_deep_merges_mappings_and_replaces_lists(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "build:\n  enabled: false\n  status: base\nitems: [a, b]\n",
        encoding="utf-8",
    )
    child.write_text(
        f"extends: {base.as_posix()}\n"
        "build:\n  status: child\n"
        "items: [c]\n",
        encoding="utf-8",
    )

    config = load_yaml_with_extends(child)

    assert config == {
        "build": {"enabled": False, "status": "child"},
        "items": ["c"],
    }


def test_yaml_extends_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(f"extends: {second.as_posix()}\n", encoding="utf-8")
    second.write_text(f"extends: {first.as_posix()}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inheritance cycle"):
        load_yaml_with_extends(first)
