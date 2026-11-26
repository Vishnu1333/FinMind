import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app import models, schemas
from backend.app.api import auth
from backend.app.db import SessionLocal


def test_registration_and_login_require_otp(monkeypatch):
    monkeypatch.setenv("OTP_HASH_SECRET", "test-secret-for-otp-hashing-32-bytes-min")
    monkeypatch.setattr(auth, "generate_otp", lambda: "123456")
    monkeypatch.setattr(auth, "send_otp_email", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.app.services.provisioning_service.ProvisioningService.provision", lambda *args, **kwargs: None)

    db: Session = SessionLocal()
    email = "otp-flow-test@example.com"
    try:
        # Ensure this test can be rerun against the isolated test database.
        existing = db.query(models.User).filter(models.User.email == email).first()
        if existing:
            db.delete(existing)
            db.commit()

        response = auth.register(
            schemas.UserCreate(email=email, password="safe-test-password", business_name="OTP Flow Test Business"),
            db,
        )
        assert "sent" in response["detail"].lower()
        assert db.query(models.User).filter(models.User.email == email).first() is None

        tokens = auth.verify_registration(auth.OTPVerify(email=email, code="123456"), db)
        user = db.query(models.User).filter(models.User.email == email).first()
        assert user is not None
        assert tokens.access_token

        login_response = auth.login(schemas.UserLogin(email=email, password="safe-test-password"), db)
        assert "sent" in login_response["detail"].lower()
        with pytest.raises(HTTPException):
            auth.verify_login(auth.OTPVerify(email=email, code="000000"), db)
        login_tokens = auth.verify_login(auth.OTPVerify(email=email, code="123456"), db)
        assert login_tokens.access_token
    finally:
        db.close()
