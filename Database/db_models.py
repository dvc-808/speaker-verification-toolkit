from pydantic import Field
from beanie import Document
from typing import List

class Speaker(Document):
    fullname: str
    ssid: str = Field(..., unique=True)
    audio_path: str
    embeddings: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "fullname": "Abdulazeez Abdulazeez Adeshina",
                "ssid": "20212344914",
                "embeddings": "[2,4442....1,4094]",
            }
        }

    class Settings:
        name = "speaker"


class Admin(Document):
    username: str = Field(..., unique=True)
    password: str

    class Settings:
        name = "admin"

__all__ = [Speaker, Admin]