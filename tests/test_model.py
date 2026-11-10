import pytest
from sklearn.metrics import f1_score

from src.predict import UNCATEGORIZED, categorize, clean_text, load_model, prepare_transactions
from src.train_classifier import load_split

import pandas as pd

MIN_MACRO_F1 = 0.85
CATEGORIES = {"Bills", "Entertainment", "Food", "Groceries", "Health", "Rent", "Shopping", "Transport", "Travel"}


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_loads_with_expected_categories(model):
    assert set(model.classes_) == CATEGORIES


def test_predictions_are_valid_categories(model):
    descriptions = ["UPI/SWIGGY*ORDER/8823", "POS 123456 DMART PUNE", "something never seen before"]
    assert set(categorize(descriptions, model=model)) <= CATEGORIES


def test_blank_descriptions_are_uncategorized(model):
    assert categorize(["", None, "12345"], model=model) == [UNCATEGORIZED] * 3


def test_macro_f1_meets_threshold(model):
    _, X_test, _, y_test = load_split()
    score = f1_score(y_test, model.predict(X_test), average="macro")
    assert score >= MIN_MACRO_F1, f"macro-F1 {score:.3f} is below {MIN_MACRO_F1}"


def test_clean_text_keeps_merchant_and_drops_noise():
    assert clean_text("UPI-TATA 1MG-tata1mg@ybl-123456789012") == "tata 1mg tata1mg"


def test_prepare_transactions_drops_unusable_rows():
    raw = pd.DataFrame(
        {
            "Date": ["2026-01-05", "not a date", "2026-01-07", "2026-01-08"],
            "Description": ["UBER", "OLA", None, "REFUND"],
            "Amount": [100, 50, 75, -20],
        }
    )
    df, dropped = prepare_transactions(raw)
    assert dropped == 2
    assert list(df["amount"]) == [100, 75]
    assert list(df["description"]) == ["UBER", ""]


def test_prepare_transactions_reports_missing_columns():
    with pytest.raises(ValueError, match="amount"):
        prepare_transactions(pd.DataFrame({"date": ["2026-01-05"], "description": ["UBER"]}))
