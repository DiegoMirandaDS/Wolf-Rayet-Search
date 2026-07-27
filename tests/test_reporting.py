from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from wr_detector.reporting.review_bundle import (
    BundleFile,
    _bundle_readme,
    build_manifest,
    sha256_file,
    verify_review_bundle,
)


def test_review_bundle_manifest_hashes_files(tmp_path: Path) -> None:
    payload = tmp_path / "evidence.txt"
    payload.write_bytes(b"wolf-rayet-review")
    item = BundleFile(
        source=payload,
        archive_path="evidence/evidence.txt",
        role="test_evidence",
    )

    manifest = build_manifest([item])

    assert manifest["file_count"] == 1
    assert manifest["uncompressed_bytes"] == len(b"wolf-rayet-review")
    assert manifest["files"] == [
        {
            "path": "evidence/evidence.txt",
            "role": "test_evidence",
            "bytes": len(b"wolf-rayet-review"),
            "sha256": hashlib.sha256(b"wolf-rayet-review").hexdigest(),
        }
    ]
    assert sha256_file(payload) == manifest["files"][0]["sha256"]


def test_review_bundle_readme_uses_minimal_safe_review_install() -> None:
    readme = _bundle_readme(
        {
            "uncompressed_bytes": 1024,
            "first_stage_run_id": "run_v3_main",
            "second_layer_run_id": "run_v3_second_layer",
        }
    )

    assert "requirements-review.txt" in readme
    assert "--require-hashes" in readme
    assert "pip install -e . --no-deps" in readme
    assert "--run-id" not in readme
    assert "second-layer model binaries" in " ".join(readme.split())
    assert "official GitHub Release" in readme


def test_verify_review_bundle_rejects_unsafe_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "unsafe.zip"
    manifest = {
        "files": [
            {
                "path": "../outside.txt",
                "sha256": hashlib.sha256(b"unsafe").hexdigest(),
            }
        ]
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../outside.txt", b"unsafe")
        archive.writestr("REVIEW_BUNDLE_MANIFEST.json", json.dumps(manifest))
        archive.writestr("README_REVIEW_BUNDLE.md", "review")

    with pytest.raises(ValueError, match="unsafe archive paths"):
        verify_review_bundle(bundle)
