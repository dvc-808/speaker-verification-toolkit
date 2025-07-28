from typing import Optional, Any
from pydantic import BaseModel

class ResponseModel(BaseModel):
    status_code : int
    data: Optional[Any]
    response_type: str