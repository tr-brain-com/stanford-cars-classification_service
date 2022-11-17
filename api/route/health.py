from fastapi import APIRouter

routes = APIRouter()

@routes.get("/health")
async def health():
    return {"status":"UP"}