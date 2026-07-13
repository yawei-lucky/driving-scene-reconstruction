"""Human control input for the simulator loop."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if not math.isfinite(value):
        raise ValueError("control values must be finite")
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class HumanControl:
    """Normalized steering, throttle, and brake input.

    Steering uses ``[-1, 1]`` and throttle and brake use ``[0, 1]``.
    Call :meth:`clamped` when input may come from an untrusted device.
    """

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def clamped(self) -> "HumanControl":
        """Return a copy constrained to the normalized control ranges."""

        return HumanControl(
            steer=_clamp(self.steer, -1.0, 1.0),
            throttle=_clamp(self.throttle, 0.0, 1.0),
            brake=_clamp(self.brake, 0.0, 1.0),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` for non-finite or out-of-range controls."""

        if not math.isfinite(self.steer):
            raise ValueError("steer must be finite")
        if not -1.0 <= self.steer <= 1.0:
            raise ValueError("steer must be between -1.0 and 1.0")
        if not math.isfinite(self.throttle):
            raise ValueError("throttle must be finite")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be between 0.0 and 1.0")
        if not math.isfinite(self.brake):
            raise ValueError("brake must be finite")
        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("brake must be between 0.0 and 1.0")
