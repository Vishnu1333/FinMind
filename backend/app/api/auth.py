from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models, schemas
from ..auth_utils import hash_password, verify_password
from ..db import get_db
from ..jwt_utils import create_access_token, create_refresh_token, decode_token
from ..otp_utils import generate_otp, hash_otp, verify_otp
from ..services.email_service import send_otp_email

router = APIRouter()
oauth_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

OTP_TTL_MINUTES = 5
OTP_RESEND_SECONDS = 60
OTP_MAX_ATTEMPTS = 5
OTP_MAX_SENDS_PER_HOUR = 5


class OTPVerify(BaseModel):
    email: EmailStr
    code: str = Field(min_length=6, max_length=6, pattern=r"^[0-9]{6}$")


class OTPResend(BaseModel):
    email: EmailStr
    purpose: Literal["registration", "login"]


def _get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email.lower()).first()


def _get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def _latest_challenge(db: Session, email: str, purpose: str):
    return (
        db.query(models.OTPChallenge)
        .filter(
            models.OTPChallenge.email == email.lower(),
            models.OTPChallenge.purpose == purpose,
            models.OTPChallenge.consumed_at.is_(None),
        )
        .order_by(models.OTPChallenge.id.desc())
        .first()
    )


def _create_and_email_challenge(
    db: Session,
    *,
    email: str,
    purpose: str,
    user_id: int | None = None,
    business_name: str | None = None,
    hashed_password: str | None = None,
):
    email = email.lower()
    now = datetime.utcnow()
    latest = _latest_challenge(db, email, purpose)
    if latest and latest.resend_after and latest.resend_after > now:
        wait = max(1, int((latest.resend_after - now).total_seconds()))
        raise HTTPException(status_code=429, detail=f"Please wait {wait} seconds before requesting another code.")

    recent_count = (
        db.query(models.OTPChallenge)
        .filter(
            models.OTPChallenge.email == email,
            models.OTPChallenge.purpose == purpose,
            models.OTPChallenge.created_at >= now - timedelta(hours=1),
        )
        .count()
    )
    if recent_count >= OTP_MAX_SENDS_PER_HOUR:
        raise HTTPException(status_code=429, detail="Too many verification emails. Please try again later.")

    code = generate_otp()
    challenge = models.OTPChallenge(
        email=email,
        purpose=purpose,
        user_id=user_id,
        business_name=business_name,
        hashed_password=hashed_password,
        otp_hash=hash_otp(email, purpose, code),
        expires_at=now + timedelta(minutes=OTP_TTL_MINUTES),
        attempts=0,
        resend_after=now + timedelta(seconds=OTP_RESEND_SECONDS),
    )
    db.add(challenge)
    try:
        send_otp_email(email, code, purpose)
        db.commit()
        db.refresh(challenge)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to send verification email. Check the FinMind SMTP configuration and try again.",
        )
    return {
        "detail": "A verification code has been sent to your email.",
        "expires_in_seconds": OTP_TTL_MINUTES * 60,
        "resend_after_seconds": OTP_RESEND_SECONDS,
    }


def _validate_challenge(db: Session, email: str, purpose: str, code: str):
    challenge = _latest_challenge(db, email, purpose)
    if not challenge:
        raise HTTPException(status_code=400, detail="No active verification request. Request a new code.")
    now = datetime.utcnow()
    if challenge.expires_at <= now:
        challenge.consumed_at = now
        db.commit()
        raise HTTPException(status_code=400, detail="Verification code expired. Request a new code.")
    if challenge.attempts >= OTP_MAX_ATTEMPTS:
        challenge.consumed_at = now
        db.commit()
        raise HTTPException(status_code=429, detail="Too many incorrect codes. Request a new code.")
    if not verify_otp(email.lower(), purpose, code, challenge.otp_hash):
        challenge.attempts += 1
        if challenge.attempts >= OTP_MAX_ATTEMPTS:
            challenge.consumed_at = now
        db.commit()
        raise HTTPException(status_code=400, detail="Incorrect verification code.")
    return challenge


def _tokens_for_user(user: models.User):
    return schemas.Token(
        access_token=create_access_token({"sub": str(user.id), "business_id": user.business_id}),
        refresh_token=create_refresh_token(user.id),
    )


def get_current_user(token: str = Depends(oauth_scheme), db: Session = Depends(get_db)):
    try:
        payload = decode_token(token)
        if payload.get("type") == "refresh":
            raise ValueError("refresh token")
        user_id = int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    user = _get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@router.post("/register")
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    email = user_in.email.lower()
    business_name = user_in.business_name.strip()
    if not business_name:
        raise HTTPException(status_code=422, detail="Business name is required.")
    if _get_user_by_email(db, email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    if db.query(models.Business).filter(models.Business.name == business_name).first():
        raise HTTPException(status_code=409, detail="A business with this name already exists.")
    return _create_and_email_challenge(
        db,
        email=email,
        purpose="registration",
        business_name=business_name,
        hashed_password=hash_password(user_in.password),
    )


@router.post("/register/verify", response_model=schemas.Token)
def verify_registration(payload: OTPVerify, db: Session = Depends(get_db)):
    email = payload.email.lower()
    challenge = _validate_challenge(db, email, "registration", payload.code)
    if not challenge.business_name or not challenge.hashed_password:
        raise HTTPException(status_code=400, detail="Registration request is invalid. Start again.")
    if _get_user_by_email(db, email):
        challenge.consumed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    if db.query(models.Business).filter(models.Business.name == challenge.business_name).first():
        challenge.consumed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=409, detail="A business with this name already exists.")

    business = models.Business(name=challenge.business_name)
    db.add(business)
    db.flush()
    user = models.User(email=email, hashed_password=challenge.hashed_password, business_id=business.id)
    db.add(user)
    challenge.consumed_at = datetime.utcnow()
    try:
        db.commit()
        db.refresh(user)
        db.refresh(business)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account or business with these details already exists.")

    from ..services.provisioning_service import ProvisioningService
    ProvisioningService(db).provision(business, user)
    return _tokens_for_user(user)


@router.post("/login")
def login(user_in: schemas.UserLogin, db: Session = Depends(get_db)):
    email = user_in.email.lower()
    user = _get_user_by_email(db, email)
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    return _create_and_email_challenge(db, email=email, purpose="login", user_id=user.id)


@router.post("/login/verify", response_model=schemas.Token)
def verify_login(payload: OTPVerify, db: Session = Depends(get_db)):
    email = payload.email.lower()
    challenge = _validate_challenge(db, email, "login", payload.code)
    if not challenge.user_id:
        raise HTTPException(status_code=400, detail="Login verification request is invalid. Sign in again.")
    user = _get_user_by_id(db, challenge.user_id)
    if not user or user.email.lower() != email:
        challenge.consumed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=401, detail="Account could not be verified.")
    challenge.consumed_at = datetime.utcnow()
    db.commit()
    return _tokens_for_user(user)


@router.post("/otp/resend")
def resend_otp(payload: OTPResend, db: Session = Depends(get_db)):
    email = payload.email.lower()
    challenge = _latest_challenge(db, email, payload.purpose)
    if not challenge:
        raise HTTPException(status_code=400, detail="No active verification request. Start again.")
    if payload.purpose == "registration":
        if not challenge.business_name or not challenge.hashed_password:
            raise HTTPException(status_code=400, detail="Registration request is invalid. Start again.")
        if _get_user_by_email(db, email):
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        return _create_and_email_challenge(
            db,
            email=email,
            purpose="registration",
            business_name=challenge.business_name,
            hashed_password=challenge.hashed_password,
        )
    if not challenge.user_id or not _get_user_by_id(db, challenge.user_id):
        raise HTTPException(status_code=400, detail="Login verification request is invalid. Sign in again.")
    return _create_and_email_challenge(db, email=email, purpose="login", user_id=challenge.user_id)


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
        if token_data.get("type") != "refresh":
            raise ValueError("not refresh token")
        user_id = int(token_data.get("sub"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    user = _get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _tokens_for_user(user)


@router.post("/logout")
def logout():
    # Stateless logout: the client discards its access/refresh tokens.
    return {"detail": "Logged out"}
