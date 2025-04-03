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

# Fungsi untuk mengubah data MongoDB ke Pydantic Model
def mongo_to_pydantic(document):
    return ScrapingResult(
        id=str(document["_id"]),
        original_title=document["original_title"],
        original_abstract=document["original_abstract"],
        cleaned_title=document["cleaned_title"],
        cleaned_abstract=document["cleaned_abstract"],
        venue=document["venue"],
        authors=document["authors"],
        num_citations=document["num_citations"],
        gsrank=document["gsrank"],
        scrape_timestamp=document["scrape_timestamp"]
    )