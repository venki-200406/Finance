"""Generate realistic, deliberately messy fake Indian bank transactions.

Run from the project root: python -m src.generate_data
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "transactions.csv"

# A fixed end date keeps the output identical from one run to the next.
END_MONTH = "2026-10"
ANOMALIES_PER_TYPE = 10


@dataclass(frozen=True)
class Merchant:
    aliases: tuple[str, ...]
    # Most merchants have one category. Several mean the same text can have different labels.
    categories: dict[str, float]
    median_amount: float
    spread: float
    weight: float


MERCHANTS = [
    Merchant(("SWIGGY", "BUNDL TECHNOLOGIES", "SWIGGY LTD"), {"Food": 1}, 380, 0.45, 13),
    Merchant(("ZOMATO", "ZOMATO LTD", "ZOMATO ONLINE"), {"Food": 1}, 420, 0.45, 12),
    Merchant(("DOMINOS", "DOMINOS PIZZA"), {"Food": 1}, 520, 0.35, 3),
    Merchant(("CAFE COFFEE DAY", "CCD"), {"Food": 1}, 260, 0.35, 2),
    Merchant(("BLINKIT", "BLINK COMMERCE"), {"Groceries": 0.85, "Food": 0.15}, 480, 0.55, 7),
    Merchant(("ZEPTO", "KIRANAKART"), {"Groceries": 0.85, "Food": 0.15}, 450, 0.55, 6),
    Merchant(("SWIGGY INSTAMART", "INSTAMART"), {"Groceries": 0.9, "Food": 0.1}, 520, 0.5, 5),
    Merchant(("BIGBASKET", "BIG BASKET", "INNOVATIVE RETAIL"), {"Groceries": 1}, 1250, 0.5, 5),
    Merchant(("DMART", "AVENUE SUPERMARTS"), {"Groceries": 1}, 1800, 0.45, 4),
    Merchant(("UBER", "UBER INDIA", "UBER TRIP"), {"Transport": 1}, 240, 0.55, 11),
    Merchant(("OLA", "OLA CABS", "ANI TECHNOLOGIES"), {"Transport": 1}, 220, 0.55, 8),
    Merchant(("RAPIDO", "ROPPEN TRANSPORTATION"), {"Transport": 1}, 95, 0.4, 5),
    Merchant(("INDIAN OIL", "IOCL", "HP PETROL PUMP"), {"Transport": 1}, 1200, 0.4, 4),
    Merchant(("BMTC", "NAMMA METRO"), {"Transport": 1}, 60, 0.35, 3),
    Merchant(
        ("AMAZON", "AMZN", "AMAZON SELLER", "AMAZON PAY"),
        {"Shopping": 0.7, "Groceries": 0.15, "Entertainment": 0.15},
        1100,
        0.8,
        9,
    ),
    Merchant(("FLIPKART", "FLIPKART INTERNET"), {"Shopping": 0.85, "Groceries": 0.15}, 1400, 0.8, 6),
    Merchant(("MYNTRA", "MYNTRA DESIGNS"), {"Shopping": 1}, 1500, 0.55, 4),
    Merchant(("AJIO", "RELIANCE RETAIL"), {"Shopping": 0.8, "Groceries": 0.2}, 1600, 0.55, 3),
    Merchant(("BOOKMYSHOW", "BIG TREE ENTERTAINMENT"), {"Entertainment": 1}, 620, 0.4, 4),
    Merchant(("PVR", "PVR INOX"), {"Entertainment": 1}, 540, 0.35, 3),
    Merchant(("APOLLO PHARMACY", "APOLLO HOSPITALS"), {"Health": 1}, 640, 0.75, 4),
    Merchant(("PHARMEASY", "API HOLDINGS"), {"Health": 1}, 720, 0.6, 3),
    Merchant(("1MG", "TATA 1MG"), {"Health": 1}, 610, 0.6, 3),
    Merchant(("IRCTC", "IRCTC ETICKET"), {"Travel": 1}, 1450, 0.6, 3),
    Merchant(("MAKEMYTRIP", "MMT"), {"Travel": 1}, 5200, 0.7, 2),
    Merchant(("INDIGO", "INTERGLOBE AVIATION"), {"Travel": 1}, 5600, 0.5, 1.5),
    Merchant(("REDBUS", "IBIBO GROUP"), {"Travel": 1}, 1100, 0.45, 2),
    Merchant(("PAYTM", "ONE97 COMMUNICATIONS"), {"Bills": 0.5, "Transport": 0.25, "Entertainment": 0.25}, 420, 0.7, 4),
]

# Payments to individuals, where the name gives little away.
PERSON_CATEGORIES = {"Groceries": 0.45, "Food": 0.3, "Health": 0.1, "Shopping": 0.15}
PERSON_WEIGHT = 9

# (description, category, median amount, spread, day-of-month range)
RECURRING = [
    ("JIO PREPAID", "Bills", 299, 0.0, (3, 8)),
    ("BESCOM", "Bills", 1450, 0.3, (8, 14)),
    ("BWSSB WATER", "Bills", 420, 0.25, (10, 16)),
    ("AIRTEL BROADBAND", "Bills", 999, 0.0, (5, 9)),
    ("NETFLIX", "Entertainment", 649, 0.0, (12, 13)),
    ("SPOTIFY", "Entertainment", 119, 0.0, (20, 21)),
    ("HOTSTAR", "Entertainment", 299, 0.0, (17, 18)),
]
RENT_AMOUNT = 18500
LANDLORD_COUNT = 6

CITIES = ("BENGALURU", "BANGALORE", "MUMBAI", "PUNE", "HYDERABAD", "CHENNAI", "DELHI")
HANDLES = ("ybl", "okaxis", "okhdfcbank", "oksbi", "ibl", "axl")

# Busy months per category, such as the Diwali shopping peak.
SEASONAL_BOOST = {
    "Shopping": {10: 1.7, 11: 1.5, 1: 1.2},
    "Travel": {4: 1.5, 5: 1.6, 12: 1.5},
    "Food": {12: 1.2, 2: 0.9},
    "Health": {7: 1.3, 8: 1.3},
}

MEAL_HOURS = (12, 13, 14, 19, 20, 21, 22, 23)
COMMUTE_HOURS = (8, 9, 10, 17, 18, 19, 20, 21)
DAY_HOURS = tuple(range(9, 23))
CATEGORY_HOURS = {"Food": MEAL_HOURS, "Transport": COMMUTE_HOURS}


def _digits(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("0123456789"), size=length))


def _typo(text: str, rng: np.random.Generator) -> str:
    """Drop or swap one character, as bank feeds and truncation do."""
    if len(text) < 5:
        return text
    i = int(rng.integers(1, len(text) - 2))
    if rng.random() < 0.5:
        return text[:i] + text[i + 1 :]
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _describe(name: str, rng: np.random.Generator) -> str:
    """Wrap a merchant name in one of the formats banks use."""
    if rng.random() < 0.07:
        name = _typo(name, rng)
    slug = name.replace(" ", "")
    handle = HANDLES[int(rng.integers(len(HANDLES)))]
    city = CITIES[int(rng.integers(len(CITIES)))]
    templates = (
        f"UPI/{slug}*ORDER/{_digits(rng, 4)}",
        f"UPI-{name}-{slug.lower()}@{handle}-{_digits(rng, 12)}",
        f"UPI/{_digits(rng, 12)}/{name}/{handle}",
        f"POS {_digits(rng, 6)} {name} {city}",
        f"{name} PAYMENT REF{_digits(rng, 8)}",
        f"ECOM/{slug}/TXN{_digits(rng, 10)}",
        f"BIL/ONL/{_digits(rng, 9)}/{name}",
    )
    text = templates[int(rng.integers(len(templates)))]
    return text.lower() if rng.random() < 0.1 else text


def _amount(median: float, spread: float, rng: np.random.Generator) -> float:
    value = median * float(np.exp(rng.normal(0, spread))) if spread else median
    # UPI payments are often whole rupees.
    return float(round(value)) if rng.random() < 0.5 else round(value, 2)


def _timestamp(month: pd.Period, day: int, hours: tuple[int, ...], rng: np.random.Generator) -> pd.Timestamp:
    hour = hours[int(rng.integers(len(hours)))]
    return month.to_timestamp() + timedelta(
        days=day - 1, hours=hour, minutes=int(rng.integers(60)), seconds=int(rng.integers(60))
    )


def _row(timestamp, description, amount, category) -> dict:
    return {"timestamp": timestamp, "description": description, "amount": amount, "category": category}


def _recurring_rows(months, landlords: list[str], rng: np.random.Generator) -> list[dict]:
    rows = []
    for position, month in enumerate(months):
        # Tenancies change, so the classifier has to learn "rent" rather than one name.
        landlord = landlords[position * len(landlords) // len(months)]
        for name, category, median, spread, (low, high) in RECURRING:
            day = int(rng.integers(low, high + 1))
            rows.append(_row(_timestamp(month, day, DAY_HOURS, rng), _describe(name, rng), _amount(median, spread, rng), category))
        day = int(rng.integers(1, 6))
        rows.append(_row(_timestamp(month, day, DAY_HOURS, rng), f"NEFT/{_digits(rng, 10)}/{landlord}/RENT", float(RENT_AMOUNT), "Rent"))
    return rows


def _pick_category(categories: dict[str, float], rng: np.random.Generator) -> str:
    names = list(categories)
    return names[int(rng.choice(len(names), p=np.array(list(categories.values()), dtype=float)))]


def _month_weights(categories: list[str], month: pd.Period, base: np.ndarray) -> np.ndarray:
    boost = np.array([SEASONAL_BOOST.get(c, {}).get(month.month, 1.0) for c in categories])
    weights = base * boost
    return weights / weights.sum()


def _everyday_rows(months, count: int, fake: Faker, rng: np.random.Generator) -> list[dict]:
    # Spending rises slowly and is lumpy, so the forecast has something real to find.
    month_factor = np.linspace(0.9, 1.15, len(months)) * rng.normal(1, 0.08, len(months))
    counts = rng.multinomial(count, month_factor / month_factor.sum())

    # Regular local vendors, so repeat names appear as they do in a real statement.
    people = [fake.name().upper().replace(".", "") for _ in range(25)]
    labels = [next(iter(m.categories)) for m in MERCHANTS] + ["Groceries"]
    base = np.array([m.weight for m in MERCHANTS] + [PERSON_WEIGHT], dtype=float)

    rows = []
    for month, n in zip(months, counts):
        probabilities = _month_weights(labels, month, base)
        for _ in range(n):
            day = int(rng.integers(1, month.days_in_month + 1))
            index = int(rng.choice(len(probabilities), p=probabilities))
            if index == len(MERCHANTS):
                category = _pick_category(PERSON_CATEGORIES, rng)
                name = people[int(rng.integers(len(people)))]
                median, spread = 350, 0.8
            else:
                merchant = MERCHANTS[index]
                category = _pick_category(merchant.categories, rng)
                name = merchant.aliases[int(rng.integers(len(merchant.aliases)))]
                median, spread = merchant.median_amount, merchant.spread
            hours = CATEGORY_HOURS.get(category, DAY_HOURS)
            rows.append(_row(_timestamp(month, day, hours, rng), _describe(name, rng), _amount(median, spread, rng), category))
    return rows


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mark huge amounts and night charges in place, and append duplicate charges."""
    df = df.copy()
    df["is_anomaly"] = 0
    df["anomaly_type"] = ""

    eligible = df.index[df["category"] != "Rent"].to_numpy()
    chosen = rng.choice(eligible, size=ANOMALIES_PER_TYPE * 3, replace=False)
    huge, night, source = np.split(chosen, 3)

    # The floor sits above normal spending, so cheap merchants still produce a truly huge charge.
    scaled = df.loc[huge, "amount"].to_numpy() * rng.uniform(15, 40, len(huge))
    df.loc[huge, "amount"] = np.maximum(scaled, rng.uniform(35000, 60000, len(huge))).round(2)
    df.loc[huge, ["is_anomaly", "anomaly_type"]] = [1, "huge_amount"]

    for index in night:
        original = df.at[index, "timestamp"]
        df.at[index, "timestamp"] = original.normalize() + timedelta(
            hours=int(rng.integers(2, 5)), minutes=int(rng.integers(60)), seconds=int(rng.integers(60))
        )
    df.loc[night, ["is_anomaly", "anomaly_type"]] = [1, "night_charge"]

    duplicates = df.loc[source].copy()
    duplicates["timestamp"] += pd.to_timedelta(rng.integers(30, 300, len(duplicates)), unit="s")
    duplicates["is_anomaly"] = 1
    duplicates["anomaly_type"] = "duplicate"
    return pd.concat([df, duplicates], ignore_index=True)


def generate(rows: int = 3000, months: int = 24, seed: int = 42) -> pd.DataFrame:
    """Build a time-ordered transaction table of exactly `rows` rows."""
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    fake.seed_instance(seed)

    periods = list(pd.period_range(end=END_MONTH, periods=months, freq="M"))
    landlords = [fake.name().upper().replace(".", "") for _ in range(LANDLORD_COUNT)]
    recurring = _recurring_rows(periods, landlords, rng)
    # Duplicates are appended later, so leave room for them to land on exactly `rows`.
    everyday_count = rows - len(recurring) - ANOMALIES_PER_TYPE
    if everyday_count <= 0:
        raise ValueError(f"rows must be greater than {len(recurring) + ANOMALIES_PER_TYPE}")

    df = pd.DataFrame(recurring + _everyday_rows(periods, everyday_count, fake, rng))
    df = _inject_anomalies(df, rng)
    return df.sort_values("timestamp", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = generate(args.rows, args.months, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({int(df['is_anomaly'].sum())} anomalies) to {args.out}")


if __name__ == "__main__":
    main()
