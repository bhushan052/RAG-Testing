from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class AskRequest(BaseModel):
    query: str
    top_k: int = 3