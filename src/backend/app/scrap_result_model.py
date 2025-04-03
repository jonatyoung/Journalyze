from pydantic import BaseModel

class ScrapingResult(BaseModel):
    id: str
    original_title: str
    original_abstract: str
    cleaned_title: str
    cleaned_abstract: str
    venue: str
    authors: list[str]
    num_citations: int
    gsrank: int
    scrape_timestamp: str

    class Config:
        from_attributes = True