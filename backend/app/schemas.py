from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    business_name: str = Field(..., min_length=1, description="Name of the business the user will own")

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: int  # user id
    business_id: int
    exp: int
    iat: int
    iss: str
    aud: str
    type: Optional[str] = None

class MeResponse(BaseModel):
    id: int
    email: EmailStr
    business_id: int
    business_name: str
