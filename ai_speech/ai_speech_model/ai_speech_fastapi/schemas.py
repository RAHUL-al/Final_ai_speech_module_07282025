from pydantic import BaseModel, EmailStr, Field, model_validator
from typing import List


from pydantic import BaseModel, EmailStr


class PasswordResetRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str = Field(..., min_length=8, max_length=100)

    @model_validator(mode="after")
    def validate_password_match(self):
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self



class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class UserCreate(BaseModel):
    username: str = Field(..., max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    @model_validator(mode="after")
    def validate_password_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserOut(BaseModel):
    id: str
    username: str
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    username: str | None = Field(None, max_length=50)
    email:    EmailStr | None
    password: str | None = Field(None, min_length=8)

class Token(BaseModel):
    access_token: str
    username: str

class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordReset(BaseModel):
    token: str
    new_password: str


class GeminiRequest(BaseModel):
    student_class: str
    accent: str
    topic: str
    mood: str

class TextToSpeechRequest(BaseModel):
    text: str

class EssayOut(BaseModel):
    id: int
    student_class: str
    accent: str
    topic: str
    mood: str
    content: str

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    curriculum: str
    question: str
    subject: str
    chat_history: List[dict] = []


class ChatRequest(BaseModel):
    question: str
    subject: str
    curriculum: str
