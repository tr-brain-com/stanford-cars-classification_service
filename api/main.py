from fastapi import FastAPI

from api.route import service, health

app = FastAPI(title="Car Detection App")

app.include_router(service.routes, prefix="/api/v1", tags=["car"])

app.include_router(health.routes, prefix="/actuator", tags=["health"])
