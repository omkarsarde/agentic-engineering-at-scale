"""Compatibility facade for Chapter 5 scaling fit and uncertainty modules."""

from scaling_model import ScalingLaw, ScalingObservation, fit_scaling_law, load_observations
from scaling_uncertainty import extrapolation_interval


__all__ = [
    "ScalingLaw",
    "ScalingObservation",
    "extrapolation_interval",
    "fit_scaling_law",
    "load_observations",
]
