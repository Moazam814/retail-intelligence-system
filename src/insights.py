"""Business insights and period comparison, computed only from the loaded data."""
from __future__ import annotations

import pandas as pd

from analytics import category_performance, kpis, product_performance
from ui.formatting import pct, pkr_compact


def kpi_values(df: pd.DataFrame) -> dict:
    """Headline KPIs for a slice of data (reuses analytics.kpis)."""
    base = kpis(df)
    revenue = base["Revenue"]
    return {
        "revenue": revenue,
        "profit": base["Profit"],
        "transactions": int(len(df)),
        "avg_daily_revenue": base["Avg daily revenue"],
        "margin": (base["Profit"] / revenue) if revenue else float("nan"),
    }


def pct_change(current: float, previous: float):
    """Percentage change, or None when it cannot be computed honestly."""
    if previous is None or pd.isna(previous) or previous == 0 or pd.isna(current):
        return None
    return (current - previous) / abs(previous) * 100.0


def previous_period(df: pd.DataFrame, start, end, category: str = "All"):
    """Data for the equivalent window immediately before [start, end], or None if the
    dataset does not fully cover that window."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    data = df if category == "All" else df[df["Category"] == category]
    if data.empty or prev_start < data["Date"].min():
        return None
    prev = data[data["Date"].between(prev_start, prev_end)]
    return None if prev.empty else prev


def generate_insights(view: pd.DataFrame, inventory: pd.DataFrame | None = None) -> list[dict]:
    """Plain-language findings derived from the selected data. Each item: title, text, tone."""
    out: list[dict] = []
    if view.empty:
        return out

    cats = category_performance(view)
    total = cats["Revenue"].sum()
    if total > 0:
        top = cats.iloc[0]
        out.append({"title": "Category leader", "tone": "info",
                    "text": f"{top['Category']} contributes {pct(top['Revenue'] / total)} of revenue "
                            f"({pkr_compact(top['Revenue'])})."})

    products = product_performance(view)
    if len(products) >= 4 and total > 0:
        share = products.head(3)["Revenue"].sum() / total
        names = ", ".join(products.head(3)["Product"])
        out.append({"title": "Revenue concentration", "tone": "info",
                    "text": f"The top three products ({names}) account for {pct(share)} of revenue."})

    daily = view.groupby("Date")["Total_Sales"].sum()
    weekend = daily[daily.index.dayofweek >= 5]
    weekday = daily[daily.index.dayofweek < 5]
    if len(weekend) >= 2 and len(weekday) >= 5 and weekday.mean() > 0:
        uplift = weekend.mean() / weekday.mean() - 1
        direction = "higher" if uplift >= 0 else "lower"
        out.append({"title": "Weekend effect", "tone": "info",
                    "text": f"Average weekend revenue is {pct(abs(uplift))} {direction} than on weekdays."})

    monthly = view.assign(Month=view["Date"].dt.to_period("M")).groupby("Month")["Total_Sales"].sum()
    if len(monthly) >= 3:
        best = monthly.idxmax()
        above = monthly.max() / monthly.mean() - 1
        out.append({"title": "Peak month", "tone": "success",
                    "text": f"{best.strftime('%B %Y')} is the strongest month at {pkr_compact(monthly.max())}, "
                            f"{pct(above)} above the monthly average."})

    if len(products) >= 2:
        hi = products.sort_values("Margin_%", ascending=False).iloc[0]
        lo = products.sort_values("Margin_%").iloc[0]
        out.append({"title": "Margin spread", "tone": "warning" if lo["Margin_%"] < 15 else "info",
                    "text": f"{hi['Product']} has the highest margin ({hi['Margin_%']:.1f}%) and "
                            f"{lo['Product']} the lowest ({lo['Margin_%']:.1f}%)."})

    if inventory is not None and not inventory.empty:
        critical = inventory[inventory["Status"] == "Reorder"]
        if len(critical):
            names = ", ".join(critical["Product"].head(4))
            more = f" and {len(critical) - 4} more" if len(critical) > 4 else ""
            out.append({"title": "Stock attention", "tone": "danger",
                        "text": f"{len(critical)} product(s) are at or below their reorder point: {names}{more}."})
        else:
            out.append({"title": "Stock position", "tone": "success",
                        "text": "No product is currently at or below its reorder point."})
    return out
