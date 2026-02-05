from __future__ import annotations

import io
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import ScalingRequest
from src.models.xgb import evaluate_on_split, forecast_future_recursive
from src.scaling.policy import ScalingParams, recommend_scaling

router = APIRouter(tags=["scaling"])


def _plot_scaling(df: pd.DataFrame) -> bytes:
    fig = plt.figure()
    ax = plt.gca()

    x = pd.to_datetime(df["datetime"])
    ax.plot(x, df["hits_pred"].values, label="hits_pred")
    ax2 = ax.twinx()
    ax2.plot(x, df["recommended_servers"].values, label="recommended_servers")

    ax.set_title("Scaling Recommendation")
    ax.grid(True)

    # legends for twinx
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@router.post("/recommend-scaling")
def recommend_scaling_api(req: ScalingRequest):
    """
    Returns ZIP including:
      - scaling_recommendation.csv
      - scaling.png
    """
    # build forecast first
    if req.mode == "evaluate":
        eval_df = evaluate_on_split(
            granularity=req.granularity,
            split=req.split,
            last_n=req.last_n,
            force_train=req.force_train,
        )
        # use predicted hits; for scaling we only need hits_pred
        df_forecast = pd.DataFrame(
            {
                "datetime": eval_df["datetime"],
                "hits_pred": eval_df["hits_pred"],
            }
        )
    else:
        fut_df = forecast_future_recursive(
            granularity=req.granularity,
            horizon_steps=req.horizon_steps,
            last_n_context=req.last_n or 500,
            force_train=req.force_train,
        )
        df_forecast = pd.DataFrame(
            {
                "datetime": fut_df["datetime"],
                "hits_pred": fut_df["hits_pred"],
            }
        )

    params = ScalingParams(
        min_servers=req.min_servers,
        max_servers=req.max_servers,
        target_utilization=req.target_utilization,
        capacity_hits_per_server_per_step=req.capacity_hits_per_server_per_step,
        scale_out_consecutive=req.scale_out_consecutive,
        scale_in_consecutive=req.scale_in_consecutive,
        cooldown_steps=req.cooldown_steps,
    )

    rec_df = recommend_scaling(df_forecast, params=params)

    png = _plot_scaling(rec_df)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("scaling_recommendation.csv", rec_df.to_csv(index=False))
        z.writestr("scaling.png", png)

    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=scaling_recommendation.zip"},
    )
