"""FastAPI for NER system.

Provides a REST API for named entity recognition with single-text and batch
endpoints. The API uses whitespace tokenization and the CRF model for
inference. When no trained model is loaded, all tokens default to 'O'.

Endpoints:
  GET  /health     — Liveness check + model status
  POST /ner        — Recognize entities in a single text
  POST /ner/batch  — Recognize entities in multiple texts at once
  GET  /ner/tags   — List supported NER tag types with descriptions
"""
import logging
from collections import Counter
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import ENTITY_DESCRIPTIONS, NER_TAGS
from src.ner_model import CRFTagger
from src.data_generator import bio_to_entities

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NER System API",
    version="1.1.0",
    description="Named Entity Recognition API supporting PER, ORG, LOC, DATE, and MISC entities.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_tagger = CRFTagger()

# Maximum number of texts in a single batch request (prevents abuse)
MAX_BATCH_SIZE = 64


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text for entity recognition")


class BatchNERRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of input texts")

    @field_validator("texts")
    @classmethod
    def validate_batch_size(cls, v: List[str]) -> List[str]:
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(v)} exceeds maximum of {MAX_BATCH_SIZE}")
        if any(len(t) > 10000 for t in v):
            raise ValueError("Each text must be at most 10000 characters")
        if any(len(t.strip()) == 0 for t in v):
            raise ValueError("Empty strings are not allowed in batch")
        return v


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class NERResponse(BaseModel):
    text: str
    entities: List[Entity]
    tags: List[str]


class BatchNERResponse(BaseModel):
    results: List[NERResponse]
    summary: Dict[str, int] = Field(default_factory=dict, description="Aggregate entity type counts")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False


class TagInfo(BaseModel):
    tag: str
    description: str


def _run_ner(text: str) -> NERResponse:
    """Shared NER inference logic for single and batch endpoints."""
    tokens = text.split()
    if not tokens:
        return NERResponse(text=text, entities=[], tags=[])
    if _tagger._model is None:
        tags = ["O"] * len(tokens)
    else:
        tags = _tagger.predict([tokens])[0]
    entities = bio_to_entities(tokens, tags)
    return NERResponse(
        text=text,
        entities=[Entity(text=e["text"], type=e["type"], start=e["start"], end=e["end"]) for e in entities],
        tags=tags,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check API health and whether a trained NER model is loaded."""
    return HealthResponse(model_loaded=_tagger._model is not None)


@app.get("/ner/tags", response_model=List[TagInfo])
async def list_tags():
    """List all supported NER entity types with human-readable descriptions."""
    entity_types = sorted(set(t.split("-", 1)[1] for t in NER_TAGS if "-" in t))
    return [
        TagInfo(tag=etype, description=ENTITY_DESCRIPTIONS.get(etype, ""))
        for etype in entity_types
    ]


@app.post("/ner", response_model=NERResponse)
async def recognize_entities(req: NERRequest):
    """Recognize named entities in a single text string."""
    return _run_ner(req.text)


@app.post("/ner/batch", response_model=BatchNERResponse)
async def recognize_entities_batch(req: BatchNERRequest):
    """Recognize named entities in multiple texts.

    Returns individual results for each text plus an aggregate summary
    counting entity types across the entire batch.
    """
    results = [_run_ner(text) for text in req.texts]

    # Aggregate entity type counts across the batch
    type_counts: Counter = Counter()
    for r in results:
        for ent in r.entities:
            type_counts[ent.type] += 1

    return BatchNERResponse(results=results, summary=dict(type_counts))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
