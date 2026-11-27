import pytest

from backend.app.otp_utils import generate_otp, hash_otp, verify_otp


def test_generate_otp_is_six_digits():
    code = generate_otp()
    assert len(code) == 6
    assert code.isdigit()


def test_otp_hash_is_bound_to_email_and_purpose(monkeypatch):
    monkeypatch.setenv("OTP_HASH_SECRET", "test-secret-for-otp-hashing-32-bytes-min")
    digest = hash_otp("person@example.com", "login", "012345")
    assert verify_otp("person@example.com", "login", "012345", digest)
    assert not verify_otp("other@example.com", "login", "012345", digest)
    assert not verify_otp("person@example.com", "registration", "012345", digest)
    assert not verify_otp("person@example.com", "login", "999999", digest)


def test_hash_requires_configured_secret(monkeypatch):
    monkeypatch.delenv("OTP_HASH_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        hash_otp("person@example.com", "login", "123456")
