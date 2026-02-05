from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional


Granularity = Literal["1m", "5m", "15m"]
Split = Literal["train", "test"]


class ForecastRequest(BaseModel):
    granularity: Granularity = Field(..., description="Aggregation level: 1m, 5m, or 15m")
    split: Split = Field("test", description="Which split to predict on (default: test)")
    horizon_steps: int = Field(1, ge=1, le=60, description="How many steps ahead (in selected granularity)")
    force_train: bool = Field(False, description="Retrain model even if artifact exists")
    last_n: Optional[int] = Field(500, ge=50, le=10000, description="How many points to include in the output/plot")
