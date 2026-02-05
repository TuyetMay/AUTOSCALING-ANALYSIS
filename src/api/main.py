from fastapi import FastAPI

from src.api.routers.health import router as health_router
from src.api.routers.forecast import router as forecast_router
from src.api.routers.scaling import router as scaling_router

app = FastAPI(
    title="Autoscaling Analysis API",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(forecast_router)
app.include_router(scaling_router)


@app.get("/")
def root():
    return {"message": "OK", "docs": "/docs"}
