"""Versioned exact-variant eligibility and bitmask primitives.

This module is intentionally independent from prediction-pool storage. It owns
the stable variant-to-bit contract, union invariants, scoring eligibility and
resumable-record validation needed by a future pool build.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_VARIANT_BIT_REGISTRY: dict[str, int] = {
    "strict_photometry": 0,
    "strict_parallax_soft": 1,
    "strict_poe_1": 2,
    "strict_poe_2": 3,
    "strict_poe_3": 4,
    "strict_poe_5": 5,
    # Bits 6-7 are reserved for future strict variants.
    "relaxed_photometry": 8,
    "relaxed_parallax_soft": 9,
    "relaxed_poe_1": 10,
    "relaxed_poe_2": 11,
    "relaxed_poe_3": 12,
    "relaxed_poe_5": 13,
    # Bits 14-15 are reserved for future relaxed variants.
}


@dataclass(frozen=True)
class VariantBit:
    variant: str
    bit: int

    @property
    def value(self) -> int:
        return 1 << self.bit


@dataclass(frozen=True)
class VariantMaskSchema:
    family: str
    major_version: int
    entries: tuple[VariantBit, ...]

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "major_version": self.major_version,
            "entries": [
                {"variant": entry.variant, "bit": entry.bit, "value": entry.value}
                for entry in self.entries
            ],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    @property
    def version(self) -> str:
        return f"{self.family}_v{self.major_version}_{self.sha256[:12]}"

    def bit_for(self, variant: str) -> int:
        for entry in self.entries:
            if entry.variant == variant:
                return entry.bit
        raise KeyError(f"Variant is not registered in {self.version}: {variant}")

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "schema_sha256": self.sha256,
            **self.canonical_payload,
        }


def build_variant_mask_schema(
    variants: Sequence[str],
    *,
    registry: Mapping[str, int] = DEFAULT_VARIANT_BIT_REGISTRY,
    family: str = "compatible_variant_mask",
    major_version: int = 1,
) -> VariantMaskSchema:
    """Build an order-independent schema while preserving registry bit positions."""
    selected = set(variants)
    unknown = sorted(selected - set(registry))
    if unknown:
        raise ValueError(f"Variants have no stable bit assignment: {unknown}")
    bits = [registry[variant] for variant in selected]
    if len(bits) != len(set(bits)):
        raise ValueError("Variant bit registry contains duplicate bit positions.")
    if any(bit < 0 or bit > 63 for bit in bits):
        raise ValueError("compatible_variant_mask v1 supports bit positions 0 through 63.")
    entries = tuple(
        VariantBit(variant=variant, bit=int(registry[variant]))
        for variant in sorted(selected, key=lambda item: (registry[item], item))
    )
    return VariantMaskSchema(
        family=family,
        major_version=int(major_version),
        entries=entries,
    )


def add_compatible_variant_mask(
    frame: pd.DataFrame,
    schema: VariantMaskSchema,
    *,
    column_template: str = "exact_variant_keep__{variant}",
) -> pd.DataFrame:
    """Add the exact-union flag, count and stable uint64 variant mask."""
    out = frame.copy()
    mask = np.zeros(len(out), dtype=np.uint64)
    count = np.zeros(len(out), dtype=np.int16)
    for entry in schema.entries:
        column = column_template.format(variant=entry.variant)
        if column not in out:
            raise KeyError(f"Missing exact-variant eligibility column: {column}")
        active = out[column].fillna(False).astype(bool).to_numpy()
        mask |= active.astype(np.uint64) << np.uint64(entry.bit)
        count += active.astype(np.int16)
    out["compatible_variant_mask"] = mask
    out["compatible_variant_count"] = count
    out["passes_any_exact_variant"] = count > 0
    assert_variant_mask_invariants(out, schema, column_template=column_template)
    return out


def assert_variant_mask_invariants(
    frame: pd.DataFrame,
    schema: VariantMaskSchema,
    *,
    column_template: str = "exact_variant_keep__{variant}",
) -> None:
    """Raise when union, count, mask and per-variant booleans disagree."""
    masks = pd.to_numeric(frame["compatible_variant_mask"], errors="raise").astype("uint64")
    expected_count = pd.Series(0, index=frame.index, dtype="int16")
    for entry in schema.entries:
        column = column_template.format(variant=entry.variant)
        expected = frame[column].fillna(False).astype(bool)
        decoded = masks.map(lambda value, bit=entry.bit: bool(int(value) & (1 << bit)))
        if not decoded.equals(expected):
            raise AssertionError(f"Bit {entry.bit} does not match {column}.")
        expected_count += expected.astype("int16")
    actual_count = pd.to_numeric(frame["compatible_variant_count"], errors="raise").astype("int16")
    if not actual_count.equals(expected_count):
        raise AssertionError("compatible_variant_count does not match per-variant columns.")
    expected_union = expected_count.gt(0)
    actual_union = frame["passes_any_exact_variant"].fillna(False).astype(bool)
    if not actual_union.equals(expected_union):
        raise AssertionError("passes_any_exact_variant does not match compatible_variant_count.")


def variant_mask_active(values: pd.Series, bit: int) -> pd.Series:
    masks = pd.to_numeric(values, errors="coerce").fillna(0).astype("uint64")
    return masks.map(lambda value: bool(int(value) & (1 << int(bit))))


def filter_sources_for_model_variant(
    frame: pd.DataFrame,
    variant: str,
    schema: VariantMaskSchema,
    *,
    mask_column: str = "compatible_variant_mask",
) -> pd.DataFrame:
    """Return only sources whose stable bit is active for the model variant."""
    bit = schema.bit_for(variant)
    return frame[variant_mask_active(frame[mask_column], bit)].copy()


def variant_photometry_mask(frame: pd.DataFrame, variant: str) -> pd.Series:
    allowed = {"A"} if variant.startswith("strict_") else {"A", "B"}
    tmass = frame["tmass_quality"].astype("string")
    wise = frame["wise_quality"].astype("string")
    mask = pd.Series(True, index=frame.index)
    for position in range(3):
        mask &= tmass.str[position].isin(allowed).fillna(False)
    for position in range(2):
        mask &= wise.str[position].isin(allowed).fillna(False)
    return mask


def variant_astrometry_mask(frame: pd.DataFrame, variant: str) -> pd.Series:
    parallax = pd.to_numeric(frame["parallax"], errors="coerce")
    poe = pd.to_numeric(frame["parallax_over_error"], errors="coerce")
    if variant.endswith("_photometry"):
        return pd.Series(True, index=frame.index)
    if variant.endswith("_parallax_soft"):
        return parallax.gt(0).fillna(False)
    for threshold in [1, 2, 3, 5]:
        if variant.endswith(f"_poe_{threshold}"):
            return (parallax.gt(0) & poe.ge(threshold)).fillna(False)
    raise ValueError(f"Unsupported exact variant: {variant}")


def build_tile_count_record(
    frame: pd.DataFrame,
    *,
    pool_build_id: str,
    tile_id: str,
    schema: VariantMaskSchema,
    known_excluded_column: str | None = None,
) -> dict[str, Any]:
    accepted_by_variant = {
        entry.variant: int(
            variant_mask_active(frame["compatible_variant_mask"], entry.bit).sum()
        )
        for entry in schema.entries
    }
    union = frame["passes_any_exact_variant"].fillna(False).astype(bool)
    known_excluded = (
        frame[known_excluded_column].fillna(False).astype(bool)
        if known_excluded_column
        else pd.Series(False, index=frame.index)
    )
    record = {
        "pool_build_id": pool_build_id,
        "tile_id": tile_id,
        "acquired_pre_locus": int(len(frame)),
        "accepted_by_variant": accepted_by_variant,
        "accepted_union": int(union.sum()),
        "known_excluded": int((union & known_excluded).sum()),
        "written": int((union & ~known_excluded).sum()),
        "bitmask_schema_version": schema.version,
        "bitmask_schema_sha256": schema.sha256,
    }
    validate_tile_count_record(record)
    return record


def validate_tile_count_record(record: Mapping[str, Any]) -> None:
    acquired = int(record["acquired_pre_locus"])
    union = int(record["accepted_union"])
    written = int(record["written"])
    excluded = int(record["known_excluded"])
    per_variant = {str(key): int(value) for key, value in record["accepted_by_variant"].items()}
    if min(acquired, union, written, excluded, *per_variant.values()) < 0:
        raise ValueError("Tile lineage counts cannot be negative.")
    if union > acquired or any(value > acquired for value in per_variant.values()):
        raise ValueError("Accepted counts cannot exceed acquired_pre_locus.")
    if any(value > union for value in per_variant.values()):
        raise ValueError("A per-variant count cannot exceed accepted_union.")
    if written + excluded != union:
        raise ValueError("written + known_excluded must equal accepted_union.")


def merge_resumable_sources(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("pool_build_id", "tile_id", "source_id"),
    checksum_column: str = "source_hash_v1",
) -> pd.DataFrame:
    """Idempotently merge source rows and reject conflicting resume payloads."""
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    duplicate_groups = combined[combined.duplicated(list(key_columns), keep=False)]
    if not duplicate_groups.empty and checksum_column in combined:
        conflicts = duplicate_groups.groupby(list(key_columns), dropna=False)[checksum_column].nunique()
        if conflicts.gt(1).any():
            raise ValueError("Resume contains conflicting source hashes for the same build/tile/source key.")
    return combined.drop_duplicates(list(key_columns), keep="last").reset_index(drop=True)


def merge_resumable_tiles(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("pool_build_id", "tile_id"),
    checksum_column: str = "acquisition_sha256",
) -> pd.DataFrame:
    """Idempotently merge tile records and reject conflicting acquisitions."""
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    duplicate_groups = combined[combined.duplicated(list(key_columns), keep=False)]
    if not duplicate_groups.empty and checksum_column in combined:
        conflicts = duplicate_groups.groupby(list(key_columns), dropna=False)[checksum_column].nunique()
        if conflicts.gt(1).any():
            raise ValueError("Resume contains conflicting acquisition hashes for the same build/tile key.")
    return combined.drop_duplicates(list(key_columns), keep="last").reset_index(drop=True)
