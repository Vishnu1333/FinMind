"""Common dependencies for FastAPI routes.

Provides the `get_current_user` dependency that validates the JWT access token
and returns the authenticated ``User`` ORM object.
"""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from .jwt_utils import decode_token
from . import db as database, models
from .api.auth import oauth_scheme as _oauth_scheme


def get_current_user(token: str = Depends(_oauth_scheme), db: Session = Depends(database.get_db)) -> models.User:
    """Validate the bearer JWT and return the authenticated user.

    Rejects refresh tokens when used in place of access tokens.
    """
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    if payload.get("type") == "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not allowed here",
        )
    user_id = int(payload.get("sub"))
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
