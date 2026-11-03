"""Experimental overlap feedback for bounded periodic preparation studies."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass
class OverlapControl:
    taper_start: float = 0.004
    hold_at: float = 0.008
    resume_at: float = 0.006
    contact_rate: float = 5.0
    held: bool = False
    previous_overlap: float | None = None

    def __post_init__(self):
        if not 0 < self.taper_start < self.resume_at < self.hold_at < 0.01:
            raise ValueError("Require 0 < taper < resume < hold < original 1% guard")
        if not math.isfinite(self.contact_rate) or self.contact_rate <= 0:
            raise ValueError("Positive finite contact compression cap required")

    def command(self, requested, overlap):
        """Taper compression only; expansion remains available to relieve pressure."""
        requested = np.asarray(requested, dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError("Three finite requested rates required")
        if not math.isfinite(overlap) or overlap < 0:
            raise ValueError("Finite nonnegative overlap required")
        decreasing = self.previous_overlap is None or overlap <= self.previous_overlap
        if overlap >= self.hold_at:
            self.held = True
        elif self.held and overlap <= self.resume_at and decreasing:
            self.held = False
        factor = (
            0.0
            if self.held
            else float(
                np.clip(
                    (self.hold_at - overlap) / (self.hold_at - self.taper_start), 0, 1
                )
            )
        )
        self.previous_overlap = float(overlap)
        compression = np.maximum(np.minimum(requested, 0), -self.contact_rate) * factor
        return np.maximum(requested, 0) + compression

    def state(self):
        return asdict(self)


def fast_precontact_cap(
    original_cap,
    fast_cap,
    diagnostic_dt,
    minimum_log_gap,
    *,
    max_log_strain=0.001,
    gap_fraction=0.25,
):
    """Accelerate only with bounding-sphere clearance; otherwise use original cap.

    minimum_log_gap=min(log(periodic centre distance / sum of bounding radii)).
    The gap bound is a geometric screen for the contact-free affine fixture,
    not a proof against nonaffine motion or a substitute for device guards.
    """
    if not all(
        math.isfinite(x)
        for x in [
            original_cap,
            fast_cap,
            diagnostic_dt,
            minimum_log_gap,
            max_log_strain,
            gap_fraction,
        ]
    ):
        raise ValueError("Finite precontact inputs required")
    if (
        not 0 < original_cap <= fast_cap
        or diagnostic_dt <= 0
        or max_log_strain <= 0
        or not 0 < gap_fraction < 1
    ):
        raise ValueError("Invalid precontact bounds")
    geometric_cap = gap_fraction * max(0.0, minimum_log_gap) / diagnostic_dt
    candidate = min(fast_cap, max_log_strain / diagnostic_dt, geometric_cap)
    return max(original_cap, candidate)
