import pydantic
from beanie import Document
from typing import List

class Speaker(Document):
    fullname: str
    ssid: str
    embeddings: List[float]
    
    #Pydantic convention, just a helper, nice to have when working in a team project
    class Config:
        json_schema_extra = {
            "example": {
                "fullname": "Abdulazeez Abdulazeez Adeshina",
                "ssid": "20212344914",
                "embeddings": "[2,4442....1,4094]",
            }
        }

    #Beanie convention, tell the ODM which document in the database to map
    class Settings:
        name = "speaker"
        indexes = [
            {"key": "ssid", "unique": True}
        ]

class Admin(Document):
    username: str
    password:str

    class Settings:
        name = "admin"
        indexes = [
            {"key": "username", "unique": True}
        ]

__all__ = [Admin, Speaker]
