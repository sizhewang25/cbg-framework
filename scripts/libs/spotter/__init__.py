"""Spotter (Laki et al. 2011) pooled RTT-distance model."""

from scripts.libs.spotter.spotter_model import (
    SpotterRTTModel,
    calibrate_k,
    fit_mu_sigma,
)

__all__ = ["SpotterRTTModel", "calibrate_k", "fit_mu_sigma"]
