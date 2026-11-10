"""Text cleaning, merchant extraction and model loading shared by every entry point."""

import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.joblib"

UNKNOWN_MERCHANT = "unknown"
UNCATEGORIZED = "Uncategorized"

# Payment rails, bank UPI handles and filler words that say nothing about the merchant.
NOISE_TOKENS = frozenset(
    """
    upi pos neft imps rtgs ach nach bil onl ecom txn trf ref payment pay paid order
    purchase debit dr cr to via mandate autopay ybl ibl axl okaxis okhdfcbank okicici oksbi
    upiaxis hdfcbank icici sbi axisbank
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _is_reference(token: str) -> bool:
    """True for transaction numbers, while keeping names like '1mg'."""
    return token.isdigit() or sum(c.isdigit() for c in token) >= 3


def clean_text(description: str) -> str:
    """Reduce a raw bank description to lowercase merchant words. Blank input gives ''."""
    if pd.isna(description):
        return ""
    tokens = _TOKEN.findall(str(description).lower())
    return " ".join(t for t in tokens if t not in NOISE_TOKENS and not _is_reference(t))


def extract_merchant(description: str) -> str:
    """Return the first meaningful word of a description, used to group by merchant."""
    cleaned = clean_text(description)
    return cleaned.split(" ", 1)[0] if cleaned else UNKNOWN_MERCHANT


@lru_cache(maxsize=1)
def load_model(path: str | Path = MODEL_PATH):
    """Load the saved classifier once per process."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run: python -m src.train_classifier")
    return joblib.load(path)


def categorize(descriptions, model=None) -> list[str]:
    """Predict a category for each raw description. Blank text is left uncategorized."""
    model = load_model() if model is None else model
    cleaned = [clean_text(d) for d in descriptions]
    known = [text for text in cleaned if text]
    labels = iter(model.predict(known) if known else ())
    return [str(next(labels)) if text else UNCATEGORIZED for text in cleaned]
