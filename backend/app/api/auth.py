from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel

from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..db import get_db
from .. import models
from .. import schemas
from ..auth_utils import hash_password, verify_password
from ..jwt_utils import create_access_token, create_refresh_token, decode_token, JWT_ISSUER, JWT_AUDIENCE

router = APIRouter()

oauth_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def _get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def _get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_current_user(token: str = Depends(oauth_scheme), db: Session = Depends(get_db)):
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    if payload.get("type") == "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not allowed here")
    user_id = int(payload.get("sub"))
    user = _get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

@router.post("/register", response_model=schemas.Token)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    # Ensure email uniqueness
    if _get_user_by_email(db, user_in.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")
    # Ensure business name uniqueness
    existing_business = db.query(models.Business).filter(models.Business.name == user_in.business_name).first()
    if existing_business:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A business with this name already exists")
    # Create business
    business = models.Business(name=user_in.business_name)
    db.add(business)
    db.flush()  # obtain business.id without committing yet
    # Create user with hashed password
    user = models.User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        business_id=business.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Provision the simulated banking environment for the new business
    from ..services.provisioning_service import ProvisioningService
    ProvisioningService(db).provision(business, user)

    # Create tokens
    access_token = create_access_token({"sub": str(user.id), "business_id": user.business_id})
    refresh_token = create_refresh_token(user.id)
    return schemas.Token(access_token=access_token, refresh_token=refresh_token)

@router.post("/login", response_model=schemas.Token)
def login(user_in: schemas.UserLogin, db: Session = Depends(get_db)):
    user = _get_user_by_email(db, user_in.email)
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access_token = create_access_token({"sub": str(user.id), "business_id": user.business_id})
    refresh_token = create_refresh_token(user.id)
    return schemas.Token(access_token=access_token, refresh_token=refresh_token)

@router.get("/me", response_model=schemas.MeResponse)
def read_me(current_user: models.User = Depends(get_current_user)):
    return schemas.MeResponse(
        id=current_user.id,
        email=current_user.email,
        business_id=current_user.business_id,
        business_name=current_user.business.name,
    )

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/refresh", response_model=schemas.Token)
def refresh_token(payload: RefreshRequest, db: Session = Depends(get_db)):
    try:
        token_data = decode_token(payload.refresh_token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    if token_data.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is not a refresh token")
    user_id = int(token_data.get("sub"))
    user = _get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    access_token = create_access_token({"sub": str(user.id), "business_id": user.business_id})
    new_refresh = create_refresh_token(user.id)
    return schemas.Token(access_token=access_token, refresh_token=new_refresh)

@router.post("/logout")
def logout():
    # Stateless logout – client just discards tokens. Endpoint kept for consistency.
    return {"detail": "Logged out"}
