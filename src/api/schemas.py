from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


Granularity = Literal["1m", "5m", "15m"]
Split = Literal["train", "test"]
Mode = Literal["evaluate", "future"]


class ForecastRequest(BaseModel):
    granularity: Granularity = "1m"
    mode: Mode = "evaluate"

    # evaluate mode only
    split: Split = "test"

    # future/evaluate: how many steps ahead (future) or how many points to return (evaluate tail)
    horizon_steps: int = Field(1, ge=1, le=20000)

    # training
    force_train: bool = False

    # evaluate: tail rows
    last_n: Optional[int] = Field(500, ge=50, le=50000)


class ScalingRequest(BaseModel):
    granularity: Granularity = "5m"
    mode: Mode = "future"

    # evaluate mode only
    split: Split = "test"

    horizon_steps: int = Field(288, ge=1, le=20000)  # default 1 day for 5m
    force_train: bool = False
    last_n: Optional[int] = Field(500, ge=50, le=50000)

    # scaling params
    min_servers: int = Field(1, ge=1, le=1000)
    max_servers: int = Field(50, ge=1, le=5000)
    target_utilization: float = Field(0.7, gt=0.0, le=1.0)

    capacity_hits_per_server_per_step: float = Field(5000.0, gt=0.0)

    scale_out_consecutive: int = Field(2, ge=1, le=50)
    scale_in_consecutive: int = Field(3, ge=1, le=50)
    cooldown_steps: int = Field(3, ge=0, le=500)
