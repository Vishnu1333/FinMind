import hashlib
import hmac
import os
import secrets


def generate_otp() -> str:
    """Return a cryptographically secure, zero-padded six-digit code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _otp_secret() -> bytes:
    secret = os.getenv("OTP_HASH_SECRET")
    if not secret or len(secret) < 32:
        raise RuntimeError("OTP_HASH_SECRET must be configured with at least 32 characters.")
    return secret.encode("utf-8")


def hash_otp(email: str, purpose: str, code: str) -> str:
    message = f"{email.lower()}|{purpose}|{code}".encode("utf-8")
    return hmac.new(_otp_secret(), message, hashlib.sha256).hexdigest()


def verify_otp(email: str, purpose: str, code: str, expected_hash: str) -> bool:
    try:
        actual = hash_otp(email, purpose, code)
    except RuntimeError:
        return False
    return hmac.compare_digest(actual, expected_hash)
