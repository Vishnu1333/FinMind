import os
from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError

# Load JWT configuration from environment variables (do not expose private keys)
JWT_PRIVATE_KEY = os.getenv('JWT_PRIVATE_KEY', 'secret')
JWT_PUBLIC_KEY = os.getenv('JWT_PUBLIC_KEY', 'secret')
JWT_ISSUER = os.getenv('JWT_ISSUER', 'finmind')
JWT_AUDIENCE = os.getenv('JWT_AUDIENCE', 'finmind-api')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '15'))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', '7'))

ALGORITHM = 'HS256'


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Claims to include in the token (e.g., user_id, business_id).
        expires_delta: Optional custom expiration timedelta.
    """
    to_encode = data.copy()
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({
        "exp": expire,
        "iat": now,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
    })
    return jwt.encode(to_encode, JWT_PRIVATE_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """Create a JWT refresh token for the given user ID."""
    now = datetime.utcnow()
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "iat": now,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "type": "refresh",
    }
    return jwt.encode(to_encode, JWT_PRIVATE_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token, returning its payload.

    Raises:
        JWTError: If token is invalid, expired, or fails verification.
    """
    try:
        payload = jwt.decode(
            token,
            JWT_PUBLIC_KEY,
            algorithms=[ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        return payload
    except JWTError as e:
        raise e
