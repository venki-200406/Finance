"""Forecast next month's spend per category and backtest against a naive baseline.

Run from the project root: python -m src.forecast
"""

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive

from src.anomaly import flag_anomalies
from src.predict import ROOT

DATA_PATH = ROOT / "data" / "transactions.csv"
FORECAST_PATH = ROOT / "data" / "forecast.csv"
MIN_MONTHS = 6
BACKTEST_WINDOWS = 6
# A final month that stops this many days early is treated as still in progress.
PARTIAL_MONTH_DAYS = 3


def monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Return complete months per category as the unique_id, ds, y table statsforecast expects."""
    timestamps = pd.to_datetime(df["timestamp"])
    last = timestamps.max()
    months = timestamps.dt.to_period("M").dt.to_timestamp()
    if last.days_in_month - last.day >= PARTIAL_MONTH_DAYS:
        complete = months < last.to_period("M").to_timestamp()
        df, months = df[complete], months[complete]

    wide = df.groupby([months.rename("ds"), "category"])["amount"].sum().unstack(fill_value=0.0)
    if wide.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="MS", name="ds"), fill_value=0.0)
    long = wide.reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    return long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"], ignore_index=True)


def has_enough_history(monthly: pd.DataFrame) -> bool:
    return monthly["ds"].nunique() >= MIN_MONTHS


def _engine() -> StatsForecast:
    # Season length 1 because two years of data cannot support yearly seasonality.
    return StatsForecast(models=[AutoETS(season_length=1), Naive()], freq="MS", fallback_model=Naive(), n_jobs=1)


def forecast_next_month(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return next month's forecast and last month's actual spend for each category."""
    predictions = _engine().forecast(df=monthly, h=1)
    last_actual = monthly.sort_values("ds").groupby("unique_id")["y"].last().rename("last_month")
    table = predictions.merge(last_actual, on="unique_id")
    return (
        table.rename(columns={"unique_id": "category", "ds": "month"})
        .assign(forecast=lambda t: t["AutoETS"].clip(lower=0))
        [["category", "month", "forecast", "last_month"]]
        .sort_values("forecast", ascending=False, ignore_index=True)
    )


def backtest(monthly: pd.DataFrame, windows: int = BACKTEST_WINDOWS) -> pd.DataFrame:
    """Return one-month-ahead mean absolute error per category for both models."""
    folds = _engine().cross_validation(df=monthly, h=1, step_size=1, n_windows=windows)
    errors = folds.assign(
        forecast_mae=(folds["AutoETS"].clip(lower=0) - folds["y"]).abs(),
        baseline_mae=(folds["Naive"] - folds["y"]).abs(),
    )
    table = errors.groupby("unique_id")[["forecast_mae", "baseline_mae"]].mean().reset_index()
    return table.rename(columns={"unique_id": "category"})


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"No data at {DATA_PATH}. Run: python -m src.generate_data")
    df = flag_anomalies(pd.read_csv(DATA_PATH))
    monthly = monthly_spend(df[~df["is_flagged"]])

    next_month = forecast_next_month(monthly)
    next_month.to_csv(FORECAST_PATH, index=False)
    print(next_month.round(0).to_string(index=False))
    print(f"\nTotal forecast: {next_month['forecast'].sum():,.0f}\n")

    errors = backtest(monthly)
    print(errors.round(0).to_string(index=False))
    forecast_mae, baseline_mae = errors["forecast_mae"].mean(), errors["baseline_mae"].mean()
    verdict = "beats" if forecast_mae < baseline_mae else "does not beat"
    print(f"\nMean MAE: AutoETS {forecast_mae:,.0f} vs naive {baseline_mae:,.0f} (AutoETS {verdict} the baseline)")
    print(f"Saved forecast to {FORECAST_PATH}")


if __name__ == "__main__":
    main()
