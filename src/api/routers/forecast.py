from __future__ import annotations

import io
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # server-safe backend
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ForecastRequest
from src.models.xgb import evaluate_on_split, forecast_future_recursive

router = APIRouter(prefix="/forecast", tags=["forecast"])


def _make_plot(df_pred: pd.DataFrame, title: str) -> bytes:
    """Return PNG bytes with 2 panels: hits + bytes."""
    if df_pred.empty:
        raise ValueError("No rows to plot")

    x = pd.to_datetime(df_pred["datetime"])

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(2, 1, 1)

    # Hits
    if "hits_true" in df_pred.columns:
        ax1.plot(x, df_pred["hits_true"], label="hits_true")
    ax1.plot(x, df_pred["hits_pred"], label="hits_pred")
    ax1.set_title(f"{title} — Hits")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bytes
    ax2 = fig.add_subplot(2, 1, 2)
    if "total_bytes_true" in df_pred.columns:
        ax2.plot(x, df_pred["total_bytes_true"], label="bytes_true")
    ax2.plot(x, df_pred["total_bytes_pred"], label="bytes_pred")
    ax2.set_title(f"{title} — Total Bytes")
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
        "Returns a ZIP containing predictions.csv and forecast.png. "
        "mode=evaluate uses train/test backtest; mode=future does recursive future forecast."
    ),
)
def forecast(req: ForecastRequest):
    try:
        if getattr(req, "mode", "evaluate") == "future":
            df_pred = forecast_future_recursive(
                granularity=req.granularity,
                horizon_steps=req.horizon_steps,
                last_n_context=req.last_n or 500,
                force_train=req.force_train,
            )
        else:
            df_pred = evaluate_on_split(
                granularity=req.granularity,
                split=req.split,
                last_n=req.last_n,
                force_train=req.force_train,
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    # Make CSV bytes
    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")

    # Make PNG bytes
    title = f"XGBoost Forecast ({req.granularity}, mode={getattr(req,'mode','evaluate')}, horizon={req.horizon_steps})"
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
    fname = f"forecast_{req.granularity}_{ts}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}

    return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)
