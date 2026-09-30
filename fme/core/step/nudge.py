from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass
class NudgeConfig:
    """
    Configuration for nudging a prognostic variable towards observation.

    The prognostic state at the next timestep t+dt will be blended as:
        blended = x * model_prediction + y * obs

    Parameters:
        x: Weight for the model prediction (0 <= x <= 1).
        y: Weight for the observation / forcing (0 <= y <= 1).
        model_weight: Alias for x.
        obs_weight: Alias for y.
        timescale_hours: Effective nudging timescale tau in hours.
            Computes x = exp(-dt / tau), y = 1 - x (assuming dt = 6h unless timestep_hours is provided).
        timescale_days: Effective nudging timescale tau in days.
            Computes timescale_hours = timescale_days * 24.
        timestep_hours: Model timestep in hours (default: 6.0).
    """

    x: float | None = None
    y: float | None = None
    model_weight: float | None = None
    obs_weight: float | None = None
    timescale_hours: float | None = None
    timescale_days: float | None = None
    timestep_hours: float = 6.0

    def __post_init__(self):
        # Resolve aliases
        if self.model_weight is not None:
            if self.x is not None:
                raise ValueError("Cannot specify both 'x' and 'model_weight'.")
            self.x = float(self.model_weight)

        if self.obs_weight is not None:
            if self.y is not None:
                raise ValueError("Cannot specify both 'y' and 'obs_weight'.")
            self.y = float(self.obs_weight)

        if self.timescale_days is not None:
            if self.timescale_hours is not None:
                raise ValueError(
                    "Cannot specify both 'timescale_days' and 'timescale_hours'."
                )
            self.timescale_hours = float(self.timescale_days) * 24.0

        if self.timescale_hours is not None:
            if self.timescale_hours <= 0:
                # tau = 0 means instantaneous replacement (prescribed)
                self.x = 0.0
                self.y = 1.0
            else:
                self.x = math.exp(-self.timestep_hours / self.timescale_hours)
                self.y = 1.0 - self.x

        if self.x is None or self.y is None:
            raise ValueError(
                "Either (x, y), (model_weight, obs_weight), or timescale_hours/timescale_days "
                "must be specified for NudgeConfig."
            )

        self.x = float(self.x)
        self.y = float(self.y)
