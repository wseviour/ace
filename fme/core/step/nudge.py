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
    """

    model_weight: float | None = None
    reanalysis_weight: float | None = None
    timescale_hours: float | None = None
    timescale_days: float | None = None
    timestep_hours: float = 6.0

    def __post_init__(self):
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
