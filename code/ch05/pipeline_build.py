"""Compose extraction, curation, decontamination, and exact mixture selection."""

from __future__ import annotations

from pathlib import Path

from data_records import decontaminate, extract_wet, filter_documents
from deduplication import near_deduplicate
from experiment_fixture import EVALUATION_NEEDLE, MIXTURE_WEIGHTS
from mixture import mix_documents
from synthetic_corpus import make_holdout_text, sample_text, write_synthetic_wet


def prepare_pipeline(out_dir: Path, raw_target_bytes: int) -> dict[str, object]:
    """Run the corpus gates and return every intermediate needed for evidence."""

    wet_path = out_dir / "synthetic-crawl.wet"
    records_written = write_synthetic_wet(wet_path, target_bytes=raw_target_bytes)
    extracted = extract_wet(wet_path)
    filtered, filter_rejections = filter_documents(extracted)
    unique, cluster_rows = near_deduplicate(filtered)
    decontaminated, contamination_rejections = decontaminate(unique, [EVALUATION_NEEDLE])
    if not decontaminated:
        raise RuntimeError("no documents survive decontamination")
    mixed = None
    for mixture_size in range(min(60, len(decontaminated)), 0, -1):
        try:
            mixed = mix_documents(decontaminated, MIXTURE_WEIGHTS, total_docs=mixture_size)
            break
        except ValueError:
            continue
    if mixed is None:
        raise RuntimeError("no exact positive-weight source mixture is feasible")
    stage_rows = [
        {"stage": "raw records", "documents": records_written},
        {"stage": "extracted", "documents": len(extracted)},
        {"stage": "rights + quality", "documents": len(filtered)},
        {"stage": "near-unique", "documents": len(unique)},
        {"stage": "decontaminated", "documents": len(decontaminated)},
        {"stage": "mixture", "documents": len(mixed)},
    ]
    return {
        "wet_path": wet_path,
        "extracted": extracted,
        "filter_rejections": filter_rejections,
        "cluster_rows": cluster_rows,
        "contamination_rejections": contamination_rejections,
        "mixed": mixed,
        "stage_rows": stage_rows,
        "raw_text": sample_text(extracted, seed=31),
        "clean_text": sample_text(mixed, seed=31),
        "holdout_text": make_holdout_text(),
    }
