"""Read a Google Pay transaction statement PDF into spending rows."""

import re

import pandas as pd
import pdfplumber

# The default tolerance joins words together on these statements.
X_TOLERANCE = 1
ROUNDING_ALLOWANCE = 1.0

_MONEY = r"₹([\d,]+(?:\.\d+)?)"
_ROW = re.compile(
    rf"^(?P<date>\d{{2}} [A-Za-z]{{3}}, \d{{4}}) (?P<kind>Paid to|Received from|Self transfer to) "
    rf"(?P<name>.+?) {_MONEY}$"
)
_TIME = re.compile(r"^(\d{1,2}:\d{2} [AP]M)\b")
_TOTALS = re.compile(rf"{_MONEY} {_MONEY}$", re.MULTILINE)


def _to_amount(text: str) -> float:
    return float(text.replace(",", ""))


def parse_gpay_text(text: str) -> pd.DataFrame:
    """Return the payments in statement text as timestamp, description and amount rows.

    Money received and self transfers are left out because they are not spending.
    Raises ValueError unless the payments add up to the statement's own Sent total.
    """
    totals = _TOTALS.search(text)
    if totals is None:
        raise ValueError("This does not look like a Google Pay statement.")

    lines = [line.strip() for line in text.splitlines()]
    rows = []
    for line, next_line in zip(lines, lines[1:]):
        row = _ROW.match(line)
        if row is None or row["kind"] != "Paid to":
            continue
        time = _TIME.match(next_line)
        if time is None:
            raise ValueError(f"Found a payment with no time on {row['date']}.")
        rows.append(
            {
                "timestamp": pd.to_datetime(f"{row['date']} {time[1]}", format="%d %b, %Y %I:%M %p"),
                "description": " ".join(row["name"].split()),
                "amount": _to_amount(row[4]),
            }
        )

    df = pd.DataFrame(rows, columns=["timestamp", "description", "amount"])
    sent = _to_amount(totals[1])
    if abs(df["amount"].sum() - sent) > ROUNDING_ALLOWANCE:
        raise ValueError(f"Read ₹{df['amount'].sum():,.2f} of payments, but the statement says ₹{sent:,.2f} was sent.")
    return df


def parse_gpay_pdf(file) -> pd.DataFrame:
    """Parse a Google Pay statement from a path or an open file. Raises ValueError if unreadable."""
    try:
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text(x_tolerance=X_TOLERANCE) or "" for page in pdf.pages)
    except Exception as error:
        # pdfplumber raises many error types for damaged or password-protected files.
        raise ValueError("The PDF could not be opened. It may be damaged or password protected.") from error
    return parse_gpay_text(text)
