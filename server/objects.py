from typing import Optional
from pydantic import BaseModel
from fastapi import UploadFile
    
class Question(BaseModel):
    question: str

class SplitParams(BaseModel):
    chunk_size: Optional[int] = 700
    chunk_overlap: Optional[int] = 150