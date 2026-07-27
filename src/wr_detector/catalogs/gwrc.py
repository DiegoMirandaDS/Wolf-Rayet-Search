"""GWRC/Crowther catalogue ingestion and Gaia DR3 alias extraction."""

from __future__ import annotations

import hashlib
import re
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


VERSION_RE = re.compile(r"v\d+\.\d+\s*\([^)]*\),\s*\d+\s*WR\s*stars,\s*[A-Za-z]+\s+\d{4}", re.I)
GAIA_DR3_RE = re.compile(r"^DR3\s+(\d+)$")


@dataclass(frozen=True)
class GWRCSnapshot:
    url: str
    html_path: Path
    csv_path: Path
    fetched_at_utc: str
    sha256: str
    version: str
    row_count: int


def download_html(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Wolf-Rayet-Search/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def detect_version(html: str) -> str:
    match = VERSION_RE.search(html)
    return " ".join(match.group(0).split()) if match else "unknown"


def clean_scalar(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("\xa0", " ").replace("&nbsp", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return pd.NA if text == "" else text


def clean_gwrc_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\xa0", " ").strip() for c in out.columns]
    for col in out.columns:
        out[col] = out[col].map(clean_scalar)

    numeric_columns = [
        "ID",
        "Galactic Longitude (deg)",
        "Galactic Latitude (deg)",
        "u (WR)",
        "b (WR)",
        "v (WR)",
        "r (WR)",
        "U",
        "B",
        "V",
        "G",
        "J",
        "H",
        "K",
        "Distance (kpc)",
    ]
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_gwrc_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError("No tables found in GWRC HTML.")
    table = max(tables, key=lambda t: t.shape[0])
    return clean_gwrc_table(table)


def extract_gaia_dr3_reference(df: pd.DataFrame) -> pd.DataFrame:
    if "Alias1" not in df.columns:
        raise ValueError("GWRC table does not contain Alias1.")

    out = df.copy()
    alias = out["Alias1"].astype("string").str.strip()
    source_id = alias.str.extract(GAIA_DR3_RE, expand=False)
    out = out[source_id.notna()].copy()
    out["source_id"] = source_id[source_id.notna()].astype("int64")
    out["gaia_designation"] = "Gaia DR3 " + out["source_id"].astype("string")
    out["wr_id"] = out["WR#"].astype("string").str.strip()
    return out.reset_index(drop=True)


def fetch_gwrc_snapshot(url: str, raw_dir: Path) -> tuple[GWRCSnapshot, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_bytes = download_html(url)
    html = html_bytes.decode("utf-8", errors="replace")
    fetched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = hashlib.sha256(html_bytes).hexdigest()
    version = detect_version(html)

    html_path = raw_dir / f"gwrc_{fetched_at}.html"
    html_path.write_bytes(html_bytes)

    table = parse_gwrc_html(html)
    csv_path = raw_dir / f"gwrc_{fetched_at}.csv"
    table.to_csv(csv_path, index=False)

    snapshot = GWRCSnapshot(
        url=url,
        html_path=html_path,
        csv_path=csv_path,
        fetched_at_utc=fetched_at,
        sha256=sha,
        version=version,
        row_count=len(table),
    )
    return snapshot, table
