from pydantic import BaseModel

class Journal(BaseModel):
    title: str
    cleaned_title: str
    topic: int