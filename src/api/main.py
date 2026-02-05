from fastapi import FastAPI

from src.api.routers.health import router as health_router
from src.api.routers.forecast import router as forecast_router


app = FastAPI(
    title="Autoscaling Analysis Forecast API",
    version="0.1.0",
    description=(
        "FastAPI service to forecast traffic (hits + bytes) using XGBoost. "
        "Swagger UI available at /docs."
    ),
)

app.include_router(health_router)
app.include_router(forecast_router)
