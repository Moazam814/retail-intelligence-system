"""Sales analytics and inventory logic (pure pandas, no UI dependencies)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "retail_sales.csv"
REQUIRED_COLUMNS = {"Date", "Product_ID", "Product", "Category", "Quantity",
                    "Unit_Price", "Cost_Price", "Total_Sales", "Profit", "Stock_Remaining"}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the sales dataset."""
    if not path.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/generate_data.py")
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return df


def kpis(df: pd.DataFrame) -> dict:
    """Headline metrics for the selected slice of data."""
    return {
        "Revenue": float(df["Total_Sales"].sum()),
        "Profit": float(df["Profit"].sum()),
        "Units sold": int(df["Quantity"].sum()),
        "Products": int(df["Product_ID"].nunique()),
        "Avg daily revenue": float(df.groupby("Date")["Total_Sales"].sum().mean()),
    }


def product_performance(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("Product", as_index=False)
           .agg(Units_Sold=("Quantity", "sum"), Revenue=("Total_Sales", "sum"),
                Profit=("Profit", "sum")))
    out["Margin_%"] = (100 * out["Profit"] / out["Revenue"]).round(1)
    return out.sort_values("Revenue", ascending=False)


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("Category", as_index=False)
            .agg(Revenue=("Total_Sales", "sum"), Profit=("Profit", "sum"),
                 Units_Sold=("Quantity", "sum"))
            .sort_values("Revenue", ascending=False))


def monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    work = df.assign(Month=df["Date"].dt.to_period("M").astype(str))
    return work.groupby("Month", as_index=False).agg(
        Revenue=("Total_Sales", "sum"), Profit=("Profit", "sum"))


def inventory_snapshot(df: pd.DataFrame, forecast_summary: pd.DataFrame | None = None,
                       horizon_days: int = 14, lead_time_days: int = 7,
                       service_z: float = 1.65, lookback_days: int = 28) -> pd.DataFrame:
    """Latest stock position with a reorder point and suggested order quantity.

    reorder_point = expected_daily * lead_time + safety_stock
    safety_stock  = z * std_daily * sqrt(lead_time)      (z=1.65 ~ 95% service level)

    Expected daily demand comes from the forecast when provided, otherwise from the
    recent average. Status is derived from the reorder point, not a fixed threshold.
    """
    latest = df["Date"].max()
    recent = df[df["Date"] > latest - pd.Timedelta(days=lookback_days)]
    stats = (recent.groupby("Product_ID")["Quantity"]
             .agg(Avg_Daily_Units="mean", Std_Daily_Units="std").fillna(0.0))

    snap = (df[df["Date"] == latest]
            [["Product_ID", "Product", "Category", "Stock_Remaining"]]
            .set_index("Product_ID").join(stats))
    expected = snap["Avg_Daily_Units"]
    if forecast_summary is not None:
        fc = forecast_summary.set_index("Product_ID")["Predicted_Demand"] / horizon_days
        expected = fc.reindex(snap.index).fillna(snap["Avg_Daily_Units"])
    snap["Expected_Daily_Units"] = expected.round(2)

    snap["Safety_Stock"] = (service_z * snap["Std_Daily_Units"] * np.sqrt(lead_time_days)).round(1)
    snap["Reorder_Point"] = (expected * lead_time_days + snap["Safety_Stock"]).round(1)
    snap["Days_Of_Cover"] = (snap["Stock_Remaining"] / expected.replace(0, np.nan)).round(1)
    snap["Status"] = np.select(
        [snap["Stock_Remaining"] <= snap["Reorder_Point"],
         snap["Stock_Remaining"] <= snap["Reorder_Point"] * 1.5],
        ["Reorder", "Watch"], default="Healthy")
    target = expected * (lead_time_days + horizon_days) + snap["Safety_Stock"]
    snap["Suggested_Order_Qty"] = np.ceil((target - snap["Stock_Remaining"]).clip(lower=0)).astype(int)
    cols = ["Product", "Category", "Stock_Remaining", "Expected_Daily_Units", "Days_Of_Cover",
            "Reorder_Point", "Safety_Stock", "Status", "Suggested_Order_Qty"]
    return snap[cols].reset_index().sort_values("Days_Of_Cover")
