"""Compatibility facade for the Chapter 5 data-pipeline mechanisms."""

from data_records import (
    Document,
    FilterPolicy,
    decontaminate,
    extract_wet,
    filter_documents,
    quality_features,
    quality_score,
)
from deduplication import near_deduplicate
from mixture import measure_fertility, mix_documents


__all__ = [
    "Document",
    "FilterPolicy",
    "decontaminate",
    "extract_wet",
    "filter_documents",
    "measure_fertility",
    "mix_documents",
    "near_deduplicate",
    "quality_features",
    "quality_score",
]
