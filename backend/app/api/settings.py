from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pydantic
from typing import Dict, Any, Optional

from .. import db as database, models
from ..dependencies import get_current_user

router = APIRouter(prefix="/settings", tags=["settings"])

class SettingsUpdate(pydantic.BaseModel):
    profile_data: Dict[str, Any] = None
    security_data: Dict[str, Any] = None
    notifications_data: Dict[str, Any] = None

class SettingsResponse(pydantic.BaseModel):
    profile_data: Optional[Dict[str, Any]] = None
    security_data: Optional[Dict[str, Any]] = None
    notifications_data: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}

@router.get("", response_model=SettingsResponse)
@router.get("/", response_model=SettingsResponse, include_in_schema=False)
def get_settings(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    setting = db.query(models.UserSetting).filter(
        models.UserSetting.business_id == current_user.business_id
    ).first()
    
    if not setting:
        setting = models.UserSetting(
            business_id=current_user.business_id,
            profile_data={"name": current_user.email.split("@")[0], "language": "en"},
            security_data={"2fa": False},
            notifications_data={"email": True, "push": True}
        )
        db.add(setting)
        db.commit()
        db.refresh(setting)
        
    return setting

@router.put("", response_model=SettingsResponse)
@router.put("/", response_model=SettingsResponse, include_in_schema=False)
def update_settings(
    request: SettingsUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    setting = db.query(models.UserSetting).filter(
        models.UserSetting.business_id == current_user.business_id
    ).first()
    
    if not setting:
        setting = models.UserSetting(business_id=current_user.business_id)
        db.add(setting)
        
    if request.profile_data is not None:
        setting.profile_data = request.profile_data
    if request.security_data is not None:
        setting.security_data = request.security_data
    if request.notifications_data is not None:
        setting.notifications_data = request.notifications_data
        
    db.commit()
    db.refresh(setting)
    return setting
