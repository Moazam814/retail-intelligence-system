import numpy as np
import pandas as pd


def make_df(days: int = 200, products: int = 3, weekly: bool = True) -> pd.DataFrame:
    """Small deterministic dataset: weekday-driven demand, no noise."""
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for i in range(1, products + 1):
        for d in dates:
            qty = (5 + i + (3 if d.dayofweek >= 5 else 0)) if weekly else 5 + i
            rows.append({"Date": d, "Product_ID": f"P{i:03d}", "Product": f"Item{i}",
                         "Category": "A" if i % 2 else "B", "Quantity": qty,
                         "Unit_Price": 100, "Cost_Price": 60, "Total_Sales": qty * 100,
                         "Profit": qty * 40, "Stock_Remaining": 100})
    return pd.DataFrame(rows)
