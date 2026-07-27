"""Build the optional binary artifact bundle used by external reviewers.

The Git repository remains source-first. This command packages the bounded
databases and model artifacts needed to browse the completed experiments in the
Model Explorer without placing large binary files in Git history.

Run from the repository root:

    python -m wr_detector.reporting.review_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable
from zoneinfo import ZoneInfo

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ID = "run_v3_main"
DEFAULT_SECOND_LAYER_RUN_ID = "run_v3_second_layer"
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "wolf_rayet_search_review_bundle.zip"
SELECTED_RESULT_IDS = (
    "9e08cb4f7517ab76d5b152e7a7dc53e02c826321",
    "49957d2295543512d384313f5c352a619fac7be8",
    "2efb4b2059dac012e52b9ff3f11e17fe4995dc66",
)


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str
    role: str


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if PROJECT_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"Bundle source is outside the project: {resolved}")
    return resolved


def _relative_archive_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _add(
    files: dict[str, BundleFile],
    source: str | Path,
    *,
    role: str,
) -> None:
    path = _project_path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    archive_path = _relative_archive_path(path)
    files[archive_path] = BundleFile(path, archive_path, role)


def collect_bundle_files(
    *,
    run_id: str = DEFAULT_RUN_ID,
    second_layer_run_id: str = DEFAULT_SECOND_LAYER_RUN_ID,
    selected_result_ids: Iterable[str] = SELECTED_RESULT_IDS,
) -> list[BundleFile]:
    """Resolve the bounded app/reproduction inputs from training history."""
    files: dict[str, BundleFile] = {}
    history_db = _project_path("data/databases/training_history.duckdb")

    for database in (
        history_db,
        _project_path("data/databases/wr_reference.duckdb"),
        _project_path("data/databases/simbad_negative.duckdb"),
    ):
        _add(files, database, role="model_explorer_database")

    result_ids = tuple(dict.fromkeys(selected_result_ids))
    placeholders = ",".join("?" for _ in result_ids)
    with duckdb.connect(str(history_db), read_only=True) as con:
        selected = con.execute(
            f"""
            SELECT result_id, model_path, metadata_json_path
            FROM model_results
            WHERE run_id = ? AND result_id IN ({placeholders})
            """,
            [run_id, *result_ids],
        ).fetchall()
        found = {str(row[0]) for row in selected}
        missing = sorted(set(result_ids) - found)
        if missing:
            raise ValueError(f"Selected result_id values are missing: {missing}")
        for _result_id, model_path, metadata_path in selected:
            _add(files, model_path, role="selected_first_stage_model")
            if metadata_path:
                resolved_metadata = _project_path(metadata_path)
                if resolved_metadata.is_file():
                    _add(
                        files,
                        resolved_metadata,
                        role="selected_first_stage_metadata",
                    )

        second_layer_count = con.execute(
            """
            SELECT COUNT(*)
            FROM second_layer_results
            WHERE run_id = ?
            """,
            [second_layer_run_id],
        ).fetchone()[0]
        if not second_layer_count:
            raise ValueError(f"No second-layer rows found for {second_layer_run_id}")

        second_layer_csv = con.execute(
            """
            SELECT source_csv
            FROM second_layer_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            [second_layer_run_id],
        ).fetchone()
        if second_layer_csv and second_layer_csv[0]:
            _add(files, second_layer_csv[0], role="second_layer_results")

    for path, role in (
        (f"reports/modeling/runs/{run_id}/model_training_results.csv", "first_stage_results"),
    ):
        _add(files, path, role=role)

    return sorted(files.values(), key=lambda item: item.archive_path)


def build_manifest(files: Iterable[BundleFile]) -> dict:
    rows = []
    total_bytes = 0
    for item in files:
        size = item.source.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": item.archive_path,
                "role": item.role,
                "bytes": size,
                "sha256": sha256_file(item.source),
            }
        )
    return {
        "bundle_format_version": 2,
        "generated_at": datetime.now(ZoneInfo("America/Santiago")).isoformat(),
        "project": "Wolf-Rayet Search",
        "first_stage_run_id": DEFAULT_RUN_ID,
        "second_layer_run_id": DEFAULT_SECOND_LAYER_RUN_ID,
        "second_layer_metrics_included": True,
        "second_layer_model_artifacts_included": False,
        "second_layer_reduced_datasets_included": False,
        "selected_result_ids": list(SELECTED_RESULT_IDS),
        "file_count": len(rows),
        "uncompressed_bytes": total_bytes,
        "files": rows,
    }


def _bundle_readme(manifest: dict) -> str:
    size_mb = manifest["uncompressed_bytes"] / (1024**2)
    return f"""# Wolf-Rayet Search - review artifact bundle

This ZIP is a binary companion to the Git repository. Extract it into the
repository root, preserving paths.

It contains:

- the DuckDB experiment/reference databases used by Model Explorer;
- three representative first-stage models from `{manifest["first_stage_run_id"]}`;
- compact CSV exports of the first- and second-layer result tables.

It does not contain the prediction pool, every first-stage model, second-layer
model binaries, or the reduced datasets used to fit the second layer. The
training-history database exposes metrics and predictions for all 144
first-stage configurations and the summarized second-layer audit. Validation
Layers remains available in Model Explorer; Candidate Stack requires the
omitted second-layer artifacts and is outside this review bundle.

Uncompressed payload: {size_mb:.1f} MiB.

After extraction:

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --require-hashes -r requirements-review.txt
python -m pip install -e . --no-deps --no-build-isolation
wr-detector explore-models --config configs/models.yaml
```

Only load `.joblib` files obtained from the project's official GitHub Release.
Python model artifacts are executable serialized objects and must not be
accepted from an untrusted mirror.

Verify every extracted file against `REVIEW_BUNDLE_MANIFEST.json`. The SHA-256
of the ZIP itself is printed by the bundle builder and should be copied to the
GitHub Release notes.
"""


def verify_review_bundle(path: str | Path) -> dict[str, int]:
    """Reopen a bundle and verify safe paths, membership and file hashes."""
    bundle_path = Path(path).resolve()
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Review bundle contains duplicate archive paths.")
        unsafe = [name for name in names if _unsafe_archive_path(name)]
        if unsafe:
            raise ValueError(f"Review bundle contains unsafe archive paths: {unsafe}")

        manifest = json.loads(archive.read("REVIEW_BUNDLE_MANIFEST.json"))
        expected = {
            str(row["path"]): str(row["sha256"])
            for row in manifest["files"]
        }
        allowed = set(expected) | {
            "REVIEW_BUNDLE_MANIFEST.json",
            "README_REVIEW_BUNDLE.md",
        }
        actual = set(names)
        if actual != allowed:
            missing = sorted(allowed - actual)
            extra = sorted(actual - allowed)
            raise ValueError(
                f"Review bundle membership mismatch; missing={missing}, extra={extra}"
            )

        for name, expected_hash in expected.items():
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Review bundle hash mismatch: {name}")

    return {
        "entries": len(names),
        "manifest_files": len(expected),
    }


def _unsafe_archive_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        path.is_absolute()
        or ".." in path.parts
        or value.startswith(("/", "\\"))
        or (len(value) >= 2 and value[1] == ":")
    )


def build_review_bundle(
    output: Path = DEFAULT_OUTPUT,
    *,
    compression_level: int = 6,
) -> tuple[Path, dict]:
    files = collect_bundle_files()
    manifest = build_manifest(files)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as archive:
        for item in files:
            archive.write(item.source, item.archive_path)
        archive.writestr(
            "REVIEW_BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        archive.writestr("README_REVIEW_BUNDLE.md", _bundle_readme(manifest))
    verify_review_bundle(output)
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and hash inputs without writing the ZIP.",
    )
    args = parser.parse_args()
    if args.dry_run:
        files = collect_bundle_files()
        manifest = build_manifest(files)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    output, manifest = build_review_bundle(args.output)
    verification = verify_review_bundle(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "archive_bytes": output.stat().st_size,
                "archive_sha256": sha256_file(output),
                "file_count": manifest["file_count"],
                "uncompressed_bytes": manifest["uncompressed_bytes"],
                "verified_entries": verification["entries"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
