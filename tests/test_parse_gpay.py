import pandas as pd
import pytest

from src.parse_gpay import parse_gpay_text

STATEMENT = """Transaction statement period Sent Received
01 Jan 2026 - 31 Jan 2026 ₹1,350.50 ₹500
Date & time Transaction details Amount
02 Jan, 2026 Paid to SAMPLE  CAFE counter 2 ₹1,000.50
09:38 PM UPI Transaction ID: 111111111111
Paid by Example Bank 0000
03 Jan, 2026 Received from Alex Example ₹500
12:05 AM UPI Transaction ID: 222222222222
Paid to Example Bank 0000
04 Jan, 2026 Self transfer to Example Bank 9999 ₹900
02:00 PM UPI Transaction ID: 333333333333
Paid by Example Bank 0000
05 Jan, 2026 Paid to TEST STORE ₹350
12:30 PM UPI Transaction ID: 444444444444
Paid by Example Bank 0000
Page 1 of 1
"""


def test_keeps_only_payments():
    df = parse_gpay_text(STATEMENT)
    assert list(df["description"]) == ["SAMPLE CAFE counter 2", "TEST STORE"]
    assert list(df["amount"]) == [1000.50, 350.0]


def test_reads_twelve_hour_times():
    df = parse_gpay_text(STATEMENT)
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-01-02 21:38")
    assert df["timestamp"].iloc[1] == pd.Timestamp("2026-01-05 12:30")


def test_rejects_totals_that_do_not_match():
    with pytest.raises(ValueError, match="statement says"):
        parse_gpay_text(STATEMENT.replace("₹1,350.50 ₹500", "₹9,999.00 ₹500"))


def test_rejects_other_documents():
    with pytest.raises(ValueError, match="Google Pay"):
        parse_gpay_text("Some unrelated document")
