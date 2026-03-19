"""FastAPI for NER system."""
import logging
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ner_model import CRFTagger
from src.data_generator import bio_to_entities

logger = logging.getLogger(__name__)

app = FastAPI(title="NER System API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_tagger = CRFTagger()


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class NERResponse(BaseModel):
    text: str
    entities: List[Entity]
    tags: List[str]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(model_loaded=_tagger._model is not None)


@app.post("/ner", response_model=NERResponse)
async def recognize_entities(req: NERRequest):
    tokens = req.text.split()
    if _tagger._model is None:
        tags = ["O"] * len(tokens)
    else:
        tags = _tagger.predict([tokens])[0]
    entities = bio_to_entities(tokens, tags)
    return NERResponse(
        text=req.text,
        entities=[Entity(text=e["text"], type=e["type"], start=e["start"], end=e["end"]) for e in entities],
        tags=tags,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
