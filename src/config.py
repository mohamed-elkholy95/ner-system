"""NER System — Configuration."""
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, DATA_DIR / "raw", MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# BIO (Beginning-Inside-Outside) tagging scheme:
#   B-<TYPE>  = first token of a named entity of type <TYPE>
#   I-<TYPE>  = continuation token inside the same entity
#   O         = token outside any entity
#
# Example:  "John Smith works at Google"
#   Tags:   B-PER I-PER   O     O  B-ORG
#
# Why BIO over IO? The B- prefix lets us distinguish adjacent entities of
# the same type. Without it, "John Smith Sarah Lee" would be one PER span.
NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-DATE", "I-DATE"]

# Human-readable descriptions for each entity type, useful for documentation
# and UI tooltips.
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "PER": "Person — names of individuals (e.g., 'John Smith', 'Dr. Maria Garcia')",
    "ORG": "Organization — companies, institutions, agencies (e.g., 'Google', 'United Nations')",
    "LOC": "Location — geographic places (e.g., 'New York', 'Silicon Valley')",
    "MISC": "Miscellaneous — nationalities, events, works of art that don't fit other types",
    "DATE": "Date — temporal expressions (e.g., 'January 2024', 'last week')",
}

TAG_COLORS = {
    "PER": "#1f77b4", "ORG": "#ff7f0e", "LOC": "#2ca02c",
    "MISC": "#d62728", "DATE": "#9467bd",
}

STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#ffffff",
}

API_HOST = "0.0.0.0"
API_PORT = 8009
