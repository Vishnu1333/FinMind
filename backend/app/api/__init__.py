from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Liveness probe for the FinMind API."""
    return {"status": "ok"}
