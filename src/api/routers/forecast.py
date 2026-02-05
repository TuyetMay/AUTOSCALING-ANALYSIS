from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # server-safe backend
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ForecastRequest
from src.models.xgb import predict_on_split


router = APIRouter(prefix="/forecast", tags=["forecast"])


def _make_plot(df_pred: pd.DataFrame, title: str) -> bytes:
    """Return a PNG bytes for hits and bytes true vs pred."""
    if df_pred.empty:
        raise ValueError("No prediction rows to plot")

    # Ensure datetime
    x = pd.to_datetime(df_pred["datetime"])

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, df_pred["y_hits_true"], label="hits_true")
    ax1.plot(x, df_pred["y_hits_pred"], label="hits_pred")
    ax1.set_title(f"{title} — Hits")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, df_pred["y_bytes_true"], label="bytes_true")
    ax2.plot(x, df_pred["y_bytes_pred"], label="bytes_pred")
    ax2.set_title(f"{title} — Bytes")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@router.post(
    "",
    summary="Forecast hits + bytes with XGBoost",
    description=(
        "Returns a ZIP that contains predictions.csv and forecast.png. "
        "If model artifacts are missing, the API will train on train split automatically."
    ),
)
def forecast(req: ForecastRequest):
    try:
        df_pred = predict_on_split(
            granularity=req.granularity,
            split=req.split,
            horizon_steps=req.horizon_steps,
            force_train=req.force_train,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    if req.last_n is not None and len(df_pred) > req.last_n:
        df_pred = df_pred.tail(req.last_n).reset_index(drop=True)

    # CSV bytes
    csv_buf = io.StringIO()
    df_pred.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # PNG bytes
    title = f"XGBoost Forecast ({req.granularity}, {req.split}, horizon={req.horizon_steps})"
    try:
        png_bytes = _make_plot(df_pred, title=title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {e}")

    # Zip it
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.csv", csv_bytes)
        zf.writestr("forecast.png", png_bytes)
    zip_buf.seek(0)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"forecast_{req.granularity}_{req.split}_h{req.horizon_steps}_{ts}.zip"

    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)
