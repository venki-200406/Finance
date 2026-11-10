"""Personal finance dashboard. Run from the project root: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly import flag_anomalies
from src.forecast import MIN_MONTHS, forecast_next_month, has_enough_history, monthly_spend
from src.generate_data import generate
from src.predict import categorize, load_model, prepare_transactions

SERIES_COLOR = "#2a78d6"
GRID_COLOR = "rgba(128, 128, 128, 0.18)"
RUPEES = "₹%{value:,.0f}"
MONEY_COLUMN = st.column_config.NumberColumn(format="₹%.0f")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def sample_transactions() -> pd.DataFrame:
    return generate()[["timestamp", "description", "amount"]]


@st.cache_data(show_spinner="Analyzing transactions...")
def analyze(df: pd.DataFrame):
    """Categorize, flag and forecast. Flagged one-offs are left out of the forecast."""
    table = flag_anomalies(df)
    table["category"] = categorize(table["description"], model=get_model())
    monthly = monthly_spend(table[~table["is_flagged"]])
    forecast = forecast_next_month(monthly) if has_enough_history(monthly) else None
    return table, forecast


def load_data() -> pd.DataFrame:
    """Return the uploaded transactions, or sample data when nothing is uploaded."""
    with st.sidebar:
        st.header("Your data")
        upload = st.file_uploader("Upload a CSV", type="csv")
        st.caption("Needs `timestamp` (or `date`), `description` and `amount` columns. Files are not stored.")

    if upload is None:
        st.info("Showing sample data. Upload a CSV in the sidebar to analyze your own.")
        return sample_transactions()

    try:
        df, dropped = prepare_transactions(pd.read_csv(upload))
    except ValueError as error:
        st.error(f"Could not read that file. {error}")
        st.stop()
    if df.empty:
        st.error("No usable rows found. Each row needs a readable date and a positive amount.")
        st.stop()
    if dropped:
        st.warning(f"Skipped {dropped} row(s) without a readable date or a positive amount.")
    return df


def style_axes(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        hoverlabel=dict(namelength=0),
    )
    return figure


def category_chart(spend: pd.Series) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=spend.values,
            y=spend.index,
            orientation="h",
            marker=dict(color=SERIES_COLOR, cornerradius=4),
            width=0.55,
            hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>",
        )
    )
    figure.update_xaxes(tickprefix="₹", tickformat="~s", gridcolor=GRID_COLOR, zeroline=False)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return style_axes(figure)


def trend_chart(spend: pd.Series) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=spend.index,
            y=spend.values,
            mode="lines+markers",
            line=dict(color=SERIES_COLOR, width=2),
            marker=dict(color=SERIES_COLOR, size=8),
            hovertemplate="%{x|%b %Y}: ₹%{y:,.0f}<extra></extra>",
        )
    )
    figure.update_xaxes(showgrid=False, tickformat="%b %y")
    figure.update_yaxes(tickprefix="₹", tickformat="~s", gridcolor=GRID_COLOR, zeroline=False, rangemode="tozero")
    return style_axes(figure)


def show_summary(table: pd.DataFrame, forecast: pd.DataFrame | None) -> None:
    columns = st.columns(4)
    columns[0].metric("Total spend", f"₹{table['amount'].sum():,.0f}")
    columns[1].metric("Transactions", f"{len(table):,}")
    columns[2].metric("Flagged", f"{int(table['is_flagged'].sum()):,}")
    columns[3].metric("Next month forecast", "Not enough data" if forecast is None else f"₹{forecast['forecast'].sum():,.0f}")


def show_charts(table: pd.DataFrame) -> None:
    by_category = table.groupby("category")["amount"].sum().sort_values(ascending=False)
    by_month = table.groupby(table["timestamp"].dt.to_period("M").dt.to_timestamp())["amount"].sum()

    left, right = st.columns(2)
    with left:
        st.subheader("Spend by category")
        st.plotly_chart(category_chart(by_category), width="stretch")
    with right:
        st.subheader("Monthly spend")
        st.plotly_chart(trend_chart(by_month), width="stretch")

    with st.expander("View chart data as tables"):
        left, right = st.columns(2)
        left.dataframe(
            by_category.rename("Spend").rename_axis("Category").reset_index(),
            hide_index=True,
            column_config={"Spend": MONEY_COLUMN},
        )
        right.dataframe(
            by_month.rename("Spend").rename_axis("Month").reset_index().assign(Month=lambda t: t["Month"].dt.strftime("%b %Y")),
            hide_index=True,
            column_config={"Spend": MONEY_COLUMN},
        )


def show_anomalies(table: pd.DataFrame) -> None:
    st.subheader("Anomaly alerts")
    flagged = table[table["is_flagged"]].sort_values("anomaly_score", ascending=False)
    if flagged.empty:
        st.success("No unusual transactions found.")
        return
    st.dataframe(
        flagged[["timestamp", "description", "amount", "category", "anomaly_reason"]],
        hide_index=True,
        width="stretch",
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, h:mm a"),
            "description": "Description",
            "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
            "category": "Category",
            "anomaly_reason": "Why it was flagged",
        },
    )


def show_forecast(forecast: pd.DataFrame | None) -> None:
    st.subheader("Next month forecast")
    if forecast is None:
        st.info(f"The forecast needs at least {MIN_MONTHS} complete months of transactions.")
        return
    st.caption(f"Estimated spend per category for {forecast['month'].iloc[0]:%B %Y}, excluding flagged transactions.")
    st.dataframe(
        forecast[["category", "forecast", "last_month"]],
        hide_index=True,
        width="stretch",
        column_config={
            "category": "Category",
            "forecast": st.column_config.NumberColumn("Forecast", format="₹%.0f"),
            "last_month": st.column_config.NumberColumn("Last month", format="₹%.0f"),
        },
    )


def show_transactions(table: pd.DataFrame) -> None:
    st.subheader("Categorized transactions")
    st.dataframe(
        table.sort_values("timestamp", ascending=False)[["timestamp", "description", "amount", "category"]],
        hide_index=True,
        width="stretch",
        height=360,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, h:mm a"),
            "description": "Description",
            "amount": st.column_config.NumberColumn("Amount", format="₹%.0f"),
            "category": "Category",
        },
    )


def main() -> None:
    st.set_page_config(page_title="Personal Finance Analyzer", page_icon="₹", layout="wide")
    st.title("Personal Finance Analyzer")

    table, forecast = analyze(load_data())
    table["timestamp"] = pd.to_datetime(table["timestamp"])

    show_summary(table, forecast)
    show_charts(table)
    show_anomalies(table)
    show_forecast(forecast)
    show_transactions(table)


main()
