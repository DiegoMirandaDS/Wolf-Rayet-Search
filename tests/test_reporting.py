from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import duckdb
import pytest

from wr_detector.reporting.review_bundle import (
    BundleFile,
    _bundle_readme,
    build_compact_negative_database,
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
    assert "app-facing projection of the SIMBAD database" in " ".join(readme.split())


def test_compact_negative_database_keeps_case_review_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.duckdb"
    output = tmp_path / "compact.duckdb"
    with duckdb.connect(str(source)) as con:
        con.execute(
            """
            CREATE TABLE simbad_negative_sources AS
            SELECT 1::BIGINT AS source_id, 'A' AS simbad_main_id,
                   'Star' AS simbad_main_type, 'O' AS simbad_sp_type,
                   'unused' AS simbad_ids
            """
        )
        con.execute(
            """
            CREATE TABLE gaia_sources AS
            SELECT 1::BIGINT AS source_id, 10.0 AS ra, -20.0 AS dec,
                   12.0 AS G, 13.0 AS BP, 11.0 AS RP, 0.2 AS parallax,
                   2.0 AS parallax_over_error, 1.1 AS ruwe,
                   999.0 AS unused_measurement
            """
        )
        con.execute(
            """
            CREATE TABLE twomass_matches AS
            SELECT 1::BIGINT AS source_id, 10.0 AS J, 9.5 AS H, 9.0 AS Ks,
                   'AAA' AS tmass_quality, 'unused' AS match_method
            """
        )
        con.execute(
            """
            CREATE TABLE wise_matches AS
            SELECT 1::BIGINT AS source_id, 8.5 AS W1, 8.0 AS W2,
                   'AA' AS wise_quality, 'unused' AS match_method
            """
        )

    build_compact_negative_database(source, output)

    with duckdb.connect(str(output), read_only=True) as con:
        assert {row[0] for row in con.execute("SHOW TABLES").fetchall()} == {
            "simbad_negative_sources",
            "gaia_sources",
            "twomass_matches",
            "wise_matches",
        }
        assert con.execute(
            "SELECT simbad_main_id, simbad_main_type, simbad_sp_type "
            "FROM simbad_negative_sources"
        ).fetchone() == ("A", "Star", "O")
        assert "unused_measurement" not in {
            row[0] for row in con.execute("DESCRIBE gaia_sources").fetchall()
        }


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
