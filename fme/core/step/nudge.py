from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass
class NudgeConfig:
    """
    Configuration for nudging a prognostic variable towards observation / reanalysis.

    The prognostic state at the next timestep t+dt will be blended as:
        blended = model_weight * model_prediction + reanalysis_weight * reanalysis

    Parameters:
        model_weight: Weight for the model prediction (0 <= model_weight <= 1).
        reanalysis_weight: Weight for the reanalysis / observation forcing (0 <= reanalysis_weight <= 1).
        timescale_hours: Effective nudging timescale tau in hours.
            Computes model_weight = exp(-dt / tau), reanalysis_weight = 1 - model_weight (assuming dt = 6h unless timestep_hours is provided).
        timescale_days: Effective nudging timescale tau in days.
            Computes timescale_hours = timescale_days * 24.
        timestep_hours: Model timestep in hours (default: 6.0).
        x: Legacy alias for model_weight.
        y: Legacy alias for reanalysis_weight.
        obs_weight: Alias for reanalysis_weight.
    """

    model_weight: float | None = None
    reanalysis_weight: float | None = None
    timescale_hours: float | None = None
    timescale_days: float | None = None
    timestep_hours: float = 6.0
    x: float | None = None
    y: float | None = None
    obs_weight: float | None = None

    def __post_init__(self):
        # Resolve aliases for model_weight
        if self.x is not None:
            if self.model_weight is not None and self.model_weight != self.x:
                raise ValueError("Conflicting values specified for 'x' and 'model_weight'.")
            self.model_weight = float(self.x)

        # Resolve aliases for reanalysis_weight
        if self.y is not None:
            if self.reanalysis_weight is not None and self.reanalysis_weight != self.y:
                raise ValueError("Conflicting values specified for 'y' and 'reanalysis_weight'.")
            self.reanalysis_weight = float(self.y)

        if self.obs_weight is not None:
            if self.reanalysis_weight is not None and self.reanalysis_weight != self.obs_weight:
                raise ValueError(
                    "Conflicting values specified for 'obs_weight' and 'reanalysis_weight'."
                )
            self.reanalysis_weight = float(self.obs_weight)

        if self.timescale_days is not None:
            if self.timescale_hours is not None:
                raise ValueError(
                    "Cannot specify both 'timescale_days' and 'timescale_hours'."
                )
            self.timescale_hours = float(self.timescale_days) * 24.0

        if self.timescale_hours is not None:
            if self.timescale_hours <= 0:
                # tau = 0 means instantaneous replacement (prescribed)
                self.model_weight = 0.0
                self.reanalysis_weight = 1.0
            else:
                self.model_weight = math.exp(-self.timestep_hours / self.timescale_hours)
                self.reanalysis_weight = 1.0 - self.model_weight

        if self.model_weight is None or self.reanalysis_weight is None:
            raise ValueError(
                "Either (model_weight, reanalysis_weight) or timescale_hours/timescale_days "
                "must be specified for NudgeConfig."
            )

        self.model_weight = float(self.model_weight)
        self.reanalysis_weight = float(self.reanalysis_weight)
        # Keep legacy aliases synchronized
        self.x = self.model_weight
        self.y = self.reanalysis_weight
