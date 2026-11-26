from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
import pydantic

from ..dependencies import get_current_user
from .. import models, db as database
from ..services.forecast_service import get_cached_forecast, InsufficientDataError
from ..services.anomaly_service import AnomalyService
from sqlalchemy.orm import Session

router = APIRouter(prefix="/predictions", tags=["predictions"])


class AnomalyCheckRequest(pydantic.BaseModel):
    amount: float
    type_code: int = 0  # retained for API compatibility; unused by the model


class AnomalyCheckResponse(pydantic.BaseModel):
    available: bool
    reason: Optional[str] = None
    is_anomaly: Optional[bool] = None
    score: Optional[float] = None


@router.get("/forecast")
def get_forecast(
    days: int = 30,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    """30-day cash-flow forecast from a model trained on THIS business's real
    transaction history. Returns model provenance; never fabricates values."""
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")
    try:
        return get_cached_forecast(db, current_user.business_id, days=days)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")


@router.get("/anomalies")
def list_anomalies(
    top_n: int = 25,
    min_amount: Optional[float] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    """Transactions flagged as unusual by an IsolationForest fitted on this
    business's own history, with concrete evidence for each flag."""
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")
    service = AnomalyService(db, current_user.business_id)
    return service.detect(top_n=top_n, min_amount=min_amount)


@router.post("/anomaly", response_model=AnomalyCheckResponse)
def check_anomaly(
    request: AnomalyCheckRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
):
    """Score a hypothetical transaction against the business's own spending
    distribution. Returns an explicit unavailability reason when there is not
    enough history – no heuristic stand-ins."""
    if not current_user.business_id:
        raise HTTPException(status_code=403, detail="User not associated with a business")
    service = AnomalyService(db, current_user.business_id)
    return AnomalyCheckResponse(**service.score_transaction(request.amount))
