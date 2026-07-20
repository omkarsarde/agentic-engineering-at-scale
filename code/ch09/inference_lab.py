"""Integrated public API and deterministic CPU build for Chapter 9.

Focused sibling modules separate request profiling, decoding mechanics, and
selective prediction while this facade preserves the chapter's import surface.
"""

from __future__ import annotations

import json
import random

from decoding_runtime import (
    distribution,
    process_logits,
    sample_index,
    sampling_probe,
    softmax,
    stream_until,
    truncate,
)
from request_profile import (
    HARDWARE,
    MODEL,
    batch_invariance_probe,
    kv_bytes,
    request_profiles,
)
from selective_prediction import (
    calibrated_score,
    calibration_metrics,
    crc_threshold,
    entropy,
    fit_temperature,
    qa_fixture,
    reliability,
    risk_curve,
    semantic_entropy_probe,
    split_conformal,
)


def run_experiment() -> dict[str, object]:
    """Run the complete deterministic chapter experiment."""
    calibration, test = qa_fixture(0), qa_fixture(7)
    tau = fit_temperature(calibration)
    threshold = crc_threshold(calibration, 0.10)
    selected = [item for item in test if float(item["confidence"]) >= threshold]
    selected_errors = sum(not bool(item["correct"]) for item in selected)
    replay_probs = distribution([4.0, 3.2, 2.2, 1.8, 0.0], 1.2, "top_p", 0.8)
    replay_a = [sample_index(replay_probs, random.Random(seed)) for seed in range(20)]
    replay_b = [sample_index(replay_probs, random.Random(seed)) for seed in range(20)]
    return {"profiles": request_profiles(), "sampling": sampling_probe(),
            "determinism": {**batch_invariance_probe(), "seed_replay_exact": replay_a == replay_b},
            "calibration": {"temperature": tau, "raw": calibration_metrics(test),
                            "calibrated": calibration_metrics(test, tau),
                            "reliability": reliability(test, tau)},
            "risk_curve": risk_curve(test),
            "crc": {"alpha": 0.10, "threshold": threshold, "coverage": len(selected) / len(test),
                    "selective_risk": selected_errors / len(selected),
                    "marginal_error": selected_errors / len(test)},
            "split_conformal": split_conformal(calibration, test, 0.10),
            "entropy": semantic_entropy_probe(),
            "streaming": {"text": stream_until([b"A\xf0", b"\x9f\x99", b"\x82ST", b"OPtail"], "STOP")}}


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))
