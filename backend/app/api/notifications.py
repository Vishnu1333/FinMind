from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import pydantic

from .. import db as database, models
from ..dependencies import get_current_user

router = APIRouter(prefix="/notifications", tags=["notifications"])

class NotificationResponse(pydantic.BaseModel):
    id: int
    title: str
    message: str
    read: bool
    severity: str
    
    model_config = {"from_attributes": True}

@router.get("", response_model=List[NotificationResponse])
@router.get("/", response_model=List[NotificationResponse], include_in_schema=False)
def get_notifications(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    notifications = db.query(models.Notification).filter(
        models.Notification.business_id == current_user.business_id
    ).order_by(models.Notification.created_at.desc()).all()
    return notifications

@router.get("/unread-count")
def get_unread_count(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    count = db.query(models.Notification).filter(
        models.Notification.business_id == current_user.business_id,
        models.Notification.read == 0
    ).count()
    return {"unread_count": count}

@router.post("/{notification_id}/read")
def mark_read(
    notification_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    notification = db.query(models.Notification).filter(
        models.Notification.id == notification_id,
        models.Notification.business_id == current_user.business_id
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
        
    notification.read = 1
    db.commit()
    return {"status": "ok"}

@router.post("/mark-all-read")
def mark_all_read(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    db.query(models.Notification).filter(
        models.Notification.business_id == current_user.business_id,
        models.Notification.read == 0
    ).update({"read": 1})
    db.commit()
    return {"status": "ok"}
