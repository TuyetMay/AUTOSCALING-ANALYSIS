from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class ScalingParams:
    min_servers: int = 1
    max_servers: int = 50
    target_utilization: float = 0.7

    # capacity per server per interval (depends on granularity)
    # example: for 5m, capacity_hits_per_server_per_step = requests/5min
    capacity_hits_per_server_per_step: float = 5000.0

    # anti-flapping
    scale_out_consecutive: int = 2
    scale_in_consecutive: int = 3
    cooldown_steps: int = 3


def recommend_scaling(
    forecast_df: pd.DataFrame,
    params: ScalingParams,
    hits_col: str = "hits_pred",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """
    Given forecast hits per step, recommend server count with:
    - min/max clamp
    - target utilization
    - consecutive breaches to scale
    - cooldown steps to prevent flapping
    """
    df = forecast_df.copy()
    if datetime_col not in df.columns:
        raise ValueError(f"Missing '{datetime_col}' in forecast_df")
    if hits_col not in df.columns:
        raise ValueError(f"Missing '{hits_col}' in forecast_df")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    min_s = int(params.min_servers)
    max_s = int(params.max_servers)

    def required_servers(hits: float) -> int:
        cap = params.capacity_hits_per_server_per_step * max(params.target_utilization, 1e-6)
        if cap <= 0:
            return max_s
        return max(1, math.ceil(max(hits, 0.0) / cap))

    current = min_s
    cooldown_left = 0
    up_streak = 0
    down_streak = 0

    servers: List[int] = []
    events: List[str] = []
    required_list: List[int] = []

    for i in range(len(df)):
        hits = float(df.loc[i, hits_col])
        req = required_servers(hits)
        req = max(min_s, min(max_s, req))
        required_list.append(req)

        event = ""

        if cooldown_left > 0:
            cooldown_left -= 1
            # during cooldown we hold
            servers.append(current)
            events.append(event)
            continue

        if req > current:
            up_streak += 1
            down_streak = 0
        elif req < current:
            down_streak += 1
            up_streak = 0
        else:
            up_streak = 0
            down_streak = 0

        # scale out
        if up_streak >= params.scale_out_consecutive:
            new_val = min(max_s, req)
            if new_val != current:
                event = f"scale_out {current}->{new_val}"
                current = new_val
                cooldown_left = params.cooldown_steps
            up_streak = 0

        # scale in
        elif down_streak >= params.scale_in_consecutive:
            new_val = max(min_s, req)
            if new_val != current:
                event = f"scale_in {current}->{new_val}"
                current = new_val
                cooldown_left = params.cooldown_steps
            down_streak = 0

        servers.append(current)
        events.append(event)

    df["required_servers"] = required_list
    df["recommended_servers"] = servers
    df["event"] = events
    return df
