"""Dashboard pages. Each page is a function taking a FilterContext."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from analytics import (category_performance, inventory_snapshot, monthly_performance,
                       product_performance)
from forecasting import FEATURES, LAGS, ROLL_WINDOWS, demand_matrix
from insights import generate_insights, kpi_values, pct_change, previous_period
from ui import charts, components as ui, html, services
from ui.formatting import (date_label, num, pct, pct_points, pkr, pkr_compact)

HORIZON = 14
Z_SCORES = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
LEVELS = {"Reorder": ("Critical", "danger"), "Watch": ("Warning", "warning"), "Healthy": ("Healthy", "success")}


@dataclass
class FilterContext:
    df: pd.DataFrame          # full dataset
    view: pd.DataFrame        # filtered by date range and category
    start: pd.Timestamp
    end: pd.Timestamp
    category: str

    @property
    def scope_chips(self) -> list[str]:
        cat = "All categories" if self.category == "All" else self.category
        return [f"{date_label(self.start)} – {date_label(self.end)}", cat]

    @property
    def category_df(self) -> pd.DataFrame:
        return self.df if self.category == "All" else self.df[self.df["Category"] == self.category]


def _no_data(ctx: FilterContext) -> bool:
    if ctx.view.empty:
        ui.empty_state("No records match the current filters",
                       "Widen the date range or choose a different category in the sidebar.")
        return True
    return False


def _auto_rule(ctx: FilterContext) -> tuple[str, str]:
    days = (ctx.end - ctx.start).days + 1
    if days <= 62:
        return "D", "Daily"
    return ("W", "Weekly") if days <= 400 else ("MS", "Monthly")


def _chart_card(title: str, description: str, fig) -> None:
    with ui.card():
        ui.section_header(title, description)
        ui.show_chart(fig)


# --------------------------------------------------------------------------- Executive Overview
def executive_overview(ctx: FilterContext) -> None:
    ui.page_header("Retail Intelligence System", "Business Performance Dashboard", ctx.scope_chips)
    if _no_data(ctx):
        return
    view = ctx.view
    cur = kpi_values(view)
    prev_df = previous_period(ctx.df, ctx.start, ctx.end, ctx.category)
    prev = kpi_values(prev_df) if prev_df is not None else None

    def d(key):
        return pct_change(cur[key], prev[key]) if prev else None

    margin_pp = ((cur["margin"] - prev["margin"]) * 100
                 if prev and not (np.isnan(cur["margin"]) or np.isnan(prev["margin"])) else None)
    ui.kpi_grid([
        {"label": "Total Revenue", "value": pkr_compact(cur["revenue"]), "help": pkr(cur["revenue"]),
         "caption": "Sales value across the selected period.", "delta": d("revenue")},
        {"label": "Total Profit", "value": pkr_compact(cur["profit"]), "help": pkr(cur["profit"]),
         "caption": "Revenue minus cost of goods sold.", "delta": d("profit")},
        {"label": "Total Transactions", "value": num(cur["transactions"]),
         "help": "Number of daily product sales records. The dataset has no receipt-level data.",
         "caption": "Daily product sales records in the period.", "delta": d("transactions")},
        {"label": "Average Daily Revenue", "value": pkr_compact(cur["avg_daily_revenue"]),
         "help": pkr(cur["avg_daily_revenue"]), "caption": "Mean total revenue per trading day.",
         "delta": d("avg_daily_revenue")},
        {"label": "Profit Margin", "value": pct(cur["margin"]), "help": "Profit divided by revenue.",
         "caption": "Share of revenue kept as profit.", "delta": margin_pp, "delta_unit": " pp"},
    ])
    if prev is None:
        ui.note("Change versus the previous period appears when the dataset contains an equivalent "
                "earlier period. Select a shorter date range (for example one quarter) to see it.")
    else:
        ui.note("Changes compare with the immediately preceding period of equal length, in the same category selection.")

    rule, gran = _auto_rule(ctx)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        _chart_card("Revenue and profit trend", f"{gran} totals in PKR", charts.revenue_profit_trend(view, rule))
    with right:
        _chart_card("Revenue by category", "Share of sales value by product category",
                    charts.category_revenue_bar(category_performance(view)))

    left, right = st.columns([2, 3], gap="medium")
    with left:
        _chart_card("Top five products", "Ranked by revenue", charts.top_products_bar(product_performance(view), 5))
    with right:
        with ui.card():
            ui.section_header("Business insights", "Generated from the selected data")
            try:
                inventory = inventory_snapshot(ctx.category_df)
            except Exception:
                inventory = None
            insights = generate_insights(view, inventory)
            if insights:
                ui.insight_list(insights)
            else:
                ui.empty_state("Not enough data for insights", "Select a longer period.")

    with ui.card():
        ui.section_header("Recent sales summary", "Latest seven trading days in the selected period")
        daily = view.groupby("Date").agg(Units=("Quantity", "sum"), Revenue=("Total_Sales", "sum"),
                                         Profit=("Profit", "sum"))
        top = view.loc[view.groupby("Date")["Total_Sales"].idxmax()].set_index("Date")["Product"]
        recent = daily.join(top.rename("Top")).tail(7).iloc[::-1].reset_index()
        recent["Margin"] = recent["Profit"] / recent["Revenue"]
        ui.table(recent, [
            {"key": "Date", "label": "Date", "fmt": lambda v: date_label(v, long=True)},
            {"key": "Units", "label": "Units sold", "align": "right", "fmt": num},
            {"key": "Revenue", "label": "Revenue", "align": "right", "fmt": pkr},
            {"key": "Profit", "label": "Profit", "align": "right", "fmt": pkr},
            {"key": "Margin", "label": "Margin", "align": "right", "fmt": pct},
            {"key": "Top", "label": "Top product"},
        ])


# --------------------------------------------------------------------------- Sales Analytics
def sales_analytics(ctx: FilterContext) -> None:
    ui.page_header("Sales Analytics", "Revenue, profit and category performance over time", ctx.scope_chips)
    if _no_data(ctx):
        return
    with ui.card():
        c1, c2 = st.columns([1, 2], gap="medium")
        gran = c1.radio("Time granularity", list(charts.RULES), index=1, horizontal=True)
        chosen = c2.multiselect("Products", sorted(ctx.view["Product"].unique()),
                                placeholder="All products in the current filters")
    data = ctx.view[ctx.view["Product"].isin(chosen)] if chosen else ctx.view
    if data.empty:
        ui.empty_state("No sales for this selection", "Choose different products or filters.")
        return
    rule = charts.RULES[gran]
    _chart_card(f"{gran} revenue and profit trend", "Revenue (blue) and profit (navy) in PKR",
                charts.revenue_profit_trend(data, rule))

    left, right = st.columns(2, gap="medium")
    with left:
        _chart_card("Monthly revenue", "Total revenue per calendar month",
                    charts.monthly_revenue_bar(monthly_performance(data)))
    with right:
        _chart_card(f"{gran} profit and margin", "Profit bars with profit margin on the right axis",
                    charts.profit_margin_trend(data, rule))

    left, right = st.columns(2, gap="medium")
    with left:
        _chart_card("Category comparison", "Revenue and profit by category",
                    charts.category_comparison(category_performance(data)))
    with right:
        _chart_card("Daily revenue distribution", "How many days fell into each revenue band",
                    charts.daily_revenue_histogram(data))

    left, right = st.columns(2, gap="medium")
    with left:
        _chart_card("Average revenue by weekday", "Weekends in navy", charts.weekday_pattern(data))
    with right:
        with ui.card():
            ui.section_header("Daily revenue statistics", "Summary of total revenue per trading day")
            daily = data.groupby("Date")["Total_Sales"].sum()
            ui.key_values([("Trading days", num(len(daily))), ("Average", pkr(daily.mean())),
                           ("Median", pkr(daily.median())), ("Lowest day", pkr(daily.min())),
                           ("Highest day", pkr(daily.max())),
                           ("Standard deviation", pkr(daily.std() if len(daily) > 1 else float("nan")))])


# --------------------------------------------------------------------------- Product Intelligence
def product_intelligence(ctx: FilterContext) -> None:
    ui.page_header("Product Intelligence", "Which products drive revenue and profit", ctx.scope_chips)
    if _no_data(ctx):
        return
    products = product_performance(ctx.view)
    cat_map = ctx.view.drop_duplicates("Product").set_index("Product")["Category"]
    products["Category"] = products["Product"].map(cat_map)

    best_rev = products.iloc[0]
    hi = products.sort_values("Margin_%", ascending=False).iloc[0]
    lo = products.sort_values("Margin_%").iloc[0]
    ui.kpi_grid([
        {"label": "Top revenue product", "value": best_rev["Product"], "caption": f"{pkr_compact(best_rev['Revenue'])} in revenue.",
         "help": "Highest total revenue in the selected period."},
        {"label": "Highest margin", "value": hi["Product"], "caption": f"{hi['Margin_%']:.1f}% profit margin.",
         "help": "Profit divided by revenue."},
        {"label": "Lowest margin", "value": lo["Product"], "caption": f"{lo['Margin_%']:.1f}% profit margin.",
         "help": "Profit divided by revenue."},
    ])

    with ui.card():
        ui.section_header("Product performance", "Top and low performers are ranked by total profit")
        c1, c2 = st.columns([2, 1], gap="medium")
        sort_label = c1.selectbox("Sort by", ["Profit", "Revenue", "Units sold", "Profit margin"])
        descending = c2.radio("Order", ["High to low", "Low to high"], horizontal=True) == "High to low"
        sort_key = {"Profit": "Profit", "Revenue": "Revenue", "Units sold": "Units_Sold", "Profit margin": "Margin_%"}[sort_label]

        ranked = products.sort_values("Profit", ascending=False).reset_index(drop=True)
        n = len(ranked)
        k = min(3, n // 3)
        ranked["Performance"] = "On track"
        if k:
            ranked.loc[:k - 1, "Performance"] = "Top performer"
            ranked.loc[n - k:, "Performance"] = "Low performer"
        shown = ranked.sort_values(sort_key, ascending=not descending)
        kinds = {"Top performer": "success", "Low performer": "danger", "On track": "neutral"}
        ui.table(shown, [
            {"key": "Product", "label": "Product"},
            {"key": "Category", "label": "Category"},
            {"key": "Units_Sold", "label": "Quantity sold", "align": "right", "fmt": num},
            {"key": "Revenue", "label": "Revenue", "align": "right", "fmt": pkr},
            {"key": "Profit", "label": "Profit", "align": "right", "fmt": pkr},
            {"key": "Margin_%", "label": "Profit margin", "align": "right", "fmt": pct_points,
             "help": "Profit divided by revenue."},
            {"key": "Performance", "label": "Performance",
             "html": lambda r: html.badge(r["Performance"], kinds[r["Performance"]])},
        ], row_class=lambda r: {"Top performer": "top", "Low performer": "low"}.get(r["Performance"], ""))
        ui.download("Download product table (CSV)", shown.to_csv(index=False).encode("utf-8"),
                    "product_performance.csv", key="dl_products")

    _chart_card("Revenue versus profit margin", "Products toward the upper right earn the most and keep the most",
                charts.margin_scatter(products))


# --------------------------------------------------------------------------- Inventory Planning
def inventory_planning(ctx: FilterContext) -> None:
    ui.page_header("Inventory Planning", "Stock position, coverage and reorder recommendations",
                   ["Latest stock snapshot", "All categories" if ctx.category == "All" else ctx.category])
    source = ctx.category_df
    forecast_result = st.session_state.get("forecast_result")

    with st.expander("Planning assumptions", expanded=False):
        c1, c2 = st.columns(2, gap="medium")
        lead_time = c1.slider("Supplier lead time (days)", 1, 21, 7,
                              help="Days between placing a replenishment order and receiving it.")
        service = c2.select_slider("Target service level", options=[0.90, 0.95, 0.99], value=0.95,
                                   format_func=lambda v: f"{v:.0%}",
                                   help="Probability of not stocking out during lead time; sets the safety stock.")
    try:
        inv = inventory_snapshot(source, forecast_summary=forecast_result[1] if forecast_result else None,
                                 horizon_days=HORIZON, lead_time_days=lead_time, service_z=Z_SCORES[service])
    except Exception as exc:  # pragma: no cover - defensive UI message
        ui.alert(f"Inventory could not be calculated: {exc}", "danger")
        return
    if inv.empty:
        ui.empty_state("No inventory data", "The dataset has no stock records for this selection.")
        return

    latest = source[source["Date"] == source["Date"].max()][["Product_ID", "Cost_Price"]]
    inv = inv.merge(latest, on="Product_ID", how="left")
    inv["Level"] = inv["Status"].map(lambda s: LEVELS[s][0])
    inv["Order_Cost"] = inv["Suggested_Order_Qty"] * inv["Cost_Price"]
    critical, warning = inv[inv["Level"] == "Critical"], inv[inv["Level"] == "Warning"]

    if len(critical):
        ui.alert(f"{len(critical)} product(s) are at or below their reorder point: "
                 f"{', '.join(critical['Product'])}. Place replenishment orders now.", "danger")
    elif len(warning):
        ui.alert(f"No critical items, but {len(warning)} product(s) are approaching their reorder point.", "warning")
    else:
        ui.alert("All monitored products are above their reorder points.", "success")

    ui.kpi_grid([
        {"label": "Products monitored", "value": num(len(inv)), "caption": "In the current category selection.",
         "help": "Products in the latest stock snapshot."},
        {"label": "Critical", "value": num(len(critical)), "caption": "At or below the reorder point.",
         "help": "Stock <= reorder point (expected demand x lead time + safety stock)."},
        {"label": "Warning", "value": num(len(warning)), "caption": "Within 50% above the reorder point.",
         "help": "Stock <= 1.5 x reorder point."},
        {"label": "Average stock coverage", "value": f"{inv['Days_Of_Cover'].mean():.1f} days",
         "caption": "Stock divided by expected daily demand.", "help": "Mean of days of cover across products."},
        {"label": "Suggested order cost", "value": pkr_compact(inv["Order_Cost"].sum()),
         "caption": f"{num(inv['Suggested_Order_Qty'].sum())} units at cost price.",
         "help": "Suggested order quantity multiplied by unit cost price."},
    ])

    max_cover = float(max(lead_time + HORIZON, inv["Days_Of_Cover"].max(skipna=True) or 1))
    with ui.card():
        ui.section_header("Inventory status", "Sorted by days of stock cover, lowest first")
        kinds = {"Critical": "danger", "Warning": "warning", "Healthy": "success"}
        ui.table(inv, [
            {"key": "Product", "label": "Product"},
            {"key": "Category", "label": "Category"},
            {"key": "Stock_Remaining", "label": "Current stock", "align": "right", "fmt": num},
            {"key": "Expected_Daily_Units", "label": "Expected daily", "align": "right",
             "fmt": lambda v: num(v, 1), "help": "Forecast demand per day when a forecast has been run, otherwise the 28-day average."},
            {"key": "Days_Of_Cover", "label": "Stock coverage",
             "html": lambda r: html.cover_bar(r["Days_Of_Cover"], max_cover, kinds[r["Level"]]),
             "help": "Estimated days until stock runs out at the expected demand rate."},
            {"key": "Reorder_Point", "label": "Reorder point", "align": "right", "fmt": lambda v: num(v, 0),
             "help": "Expected daily demand x lead time + safety stock."},
            {"key": "Safety_Stock", "label": "Safety stock", "align": "right", "fmt": lambda v: num(v, 0),
             "help": "z x standard deviation of daily demand x sqrt(lead time)."},
            {"key": "Level", "label": "Status", "html": lambda r: html.badge(r["Level"], kinds[r["Level"]])},
        ])
        demand_src = "model forecast" if forecast_result else "28-day average demand"
        ui.note(f"Expected demand source: {demand_src}. Run the Demand Forecasting page to switch to model-based demand.")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        _chart_card("Stock versus reorder point", "Bars below the diamond marker need replenishment",
                    charts.inventory_chart(inv.sort_values("Days_Of_Cover")))
    with right:
        with ui.card():
            ui.section_header("Reorder recommendations", "Covers lead time plus 14 days of demand and safety stock")
            todo = inv[(inv["Suggested_Order_Qty"] > 0)].sort_values(["Level", "Days_Of_Cover"],
                                                                      key=lambda s: s.map({"Critical": 0, "Warning": 1, "Healthy": 2}) if s.name == "Level" else s)
            if todo.empty:
                ui.empty_state("Nothing to order", "Current stock covers the planning horizon.")
            else:
                ui.table(todo, [
                    {"key": "Product", "label": "Product"},
                    {"key": "Level", "label": "Status", "html": lambda r: html.badge(r["Level"], kinds[r["Level"]])},
                    {"key": "Suggested_Order_Qty", "label": "Order qty", "align": "right", "fmt": num},
                    {"key": "Order_Cost", "label": "Est. cost", "align": "right", "fmt": pkr},
                ])
            ui.download("Download inventory plan (CSV)", inv.drop(columns=["Level"]).to_csv(index=False).encode("utf-8"),
                        "inventory_plan.csv", key="dl_inventory")


# --------------------------------------------------------------------------- Demand Forecasting
def demand_forecasting(ctx: FilterContext) -> None:
    ui.page_header("Demand Forecasting", "14-day product demand forecast with transparent model evaluation",
                   ["Random Forest", "Recursive multi-step", "Rolling-origin backtest"])
    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card():
            ui.section_header("Model information")
            ui.key_values([
                ("Algorithm", "Random Forest regressor (200 trees)"),
                ("Scope", "One global model across all products"),
                ("Target", "Units sold per product per day"),
                ("Calendar features", "Day of week, month, day of year, weekend flag"),
                ("Demand features", f"Lags {', '.join(map(str, LAGS))} days; rolling means {', '.join(map(str, ROLL_WINDOWS))} days"),
                ("Forecast method", f"Recursive, {HORIZON} days ahead"),
                ("Feature count", str(len(FEATURES)))])
    with right:
        with ui.card():
            ui.section_header("Evaluation methodology")
            ui.key_values([
                ("Scheme", "Rolling-origin backtest"),
                ("Folds", "4 folds, each forecasting 14 days"),
                ("Training", "Only data before each forecast origin"),
                ("Baselines", "Weekday mean, seasonal naive"),
                ("Metrics", "MAE, RMSE, WAPE"),
                ("Leakage control", "Features use past data only")])

    if st.button("Run forecast", type="primary", help="Trains the model, runs the backtest and forecasts the next 14 days."):
        try:
            with st.spinner("Training model, running backtest and generating forecast (about 10-20 seconds)..."):
                st.session_state["forecast_result"] = services.run_forecast(HORIZON)
                st.session_state["holdout_result"] = services.run_holdout(HORIZON)
        except Exception as exc:
            ui.alert(f"The forecast could not be generated: {exc}", "danger")
            return
    result = st.session_state.get("forecast_result")
    if result is None:
        ui.empty_state("No forecast generated yet",
                       "Select Run forecast to train the model and see accuracy metrics, charts and product forecasts.")
        return

    forecast, summary, scores = result
    rf = scores[scores["Model"] == "Random Forest"].iloc[0]
    base = scores[scores["Model"] != "Random Forest"].sort_values("MAE").iloc[0]
    gain = (base["MAE"] - rf["MAE"]) / base["MAE"]

    ui.section_header("Model evaluation", "Pooled across four 14-day backtest folds. Lower error is better.")
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("MAE", f"{rf['MAE']:.2f} units", help="Mean absolute error: average size of the daily error per product, in units.")
    m2.metric("RMSE", f"{rf['RMSE']:.2f} units", help="Root mean squared error: like MAE but penalises large misses more.")
    m3.metric("WAPE", pct(rf["WAPE"]), help="Weighted absolute percentage error: total absolute error divided by total actual demand.")
    m4.metric("MAE vs best baseline", f"{gain:+.1%}", help=f"Improvement over the best baseline ({base['Model']}). "
              "Positive means the Random Forest has lower error.")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with ui.card():
            ui.section_header("Model comparison")
            ui.table(scores, [
                {"key": "Model", "label": "Model"},
               {"key": "MAE", "label": "MAE", "align": "right", "fmt": lambda v: f"{v:.3f}"},
                {"key": "RMSE", "label": "RMSE", "align": "right", "fmt": lambda v: f"{v:.2f}"},
                {"key": "WAPE", "label": "WAPE", "align": "right", "fmt": pct}])
            ui.note("The data is synthetic with random noise, so part of the error cannot be removed by any model.")
    with right:
        _chart_card("Mean absolute error by model", "", charts.model_comparison(scores))

    with ui.card():
        ui.section_header("Actual versus predicted demand", "Product-level view of recent demand, holdout predictions and the next 14 days")
        names = summary["Product"].tolist()
        pick = st.selectbox("Product", names)
        pid = summary.loc[summary["Product"] == pick, "Product_ID"].iloc[0]
        history = demand_matrix(ctx.df)[pid].tail(60)
        holdout = st.session_state.get("holdout_result")
        pred = holdout[1][pid] if holdout is not None else pd.Series(dtype=float)
        future = forecast[forecast["Product_ID"] == pid].set_index("Date")["Predicted_Demand"]
        ui.show_chart(charts.forecast_chart(history, pred, future))
        ui.note("Holdout predictions come from a model trained without the last 14 days of actuals.")

    left, right = st.columns([2, 3], gap="medium")
    with left:
        with ui.card():
            ui.section_header(f"{pick}: next {HORIZON} days")
            table = future.rename("Predicted_Demand").reset_index()
            ui.table(table, [
                {"key": "Date", "label": "Date", "fmt": lambda v: date_label(v, long=True)},
                {"key": "Predicted_Demand", "label": "Predicted units", "align": "right", "fmt": lambda v: num(v, 1)}])
    with right:
        with ui.card():
            ui.section_header("All products: 14-day demand and projected revenue")
            ui.note(
    "This forecast covers all product categories. "
    "The sidebar category filter does not restrict the global forecasting model."
)
            price = ctx.df.sort_values("Date").groupby("Product_ID")["Unit_Price"].last()
            s = summary.copy()
            s["Projected_Revenue"] = s["Predicted_Demand"] * s["Product_ID"].map(price)
            ui.table(s, [
                {"key": "Product", "label": "Product"},
                {"key": "Predicted_Demand", "label": "Predicted units", "align": "right", "fmt": lambda v: num(v, 0)},
                {"key": "Projected_Revenue", "label": "Projected revenue", "align": "right", "fmt": pkr,
                 "help": "Predicted units multiplied by the latest unit price."}])
            ui.download("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                        "demand_forecast.csv", key="dl_forecast")


# --------------------------------------------------------------------------- Data and Methodology
def data_and_methodology(ctx: FilterContext) -> None:
    ui.page_header("Data and Methodology", "What the numbers are based on and how they are calculated")
    df = ctx.df
    ui.alert("All sales records in this project are synthetic and generated for demonstration. "
             "Results do not describe a real retailer or real-world forecast accuracy.", "info")
    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card():
            ui.section_header("Dataset summary")
            ui.key_values([("Records", num(len(df))), ("Products", num(df["Product_ID"].nunique())),
                           ("Categories", num(df["Category"].nunique())),
                           ("First date", date_label(df["Date"].min())), ("Last date", date_label(df["Date"].max())),
                           ("Trading days", num(df["Date"].nunique())), ("Source", "data/retail_sales.csv (generated)")])
    with right:
        with ui.card():
            ui.section_header("Data dictionary")
            ui.key_values([("Date", "Trading day"), ("Product_ID / Product", "Product identifier and name"),
                           ("Category", "Product category"), ("Quantity", "Units sold that day"),
                           ("Unit_Price / Cost_Price", "Selling and cost price per unit (PKR)"),
                           ("Total_Sales", "Quantity x unit price"), ("Profit", "Quantity x (price - cost)"),
                           ("Stock_Remaining", "Units in stock at end of day")])
    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card():
            ui.section_header("Forecasting methodology")
            st.markdown(
                "- A single Random Forest is trained on all products using calendar, lag and rolling-mean features.\n"
                "- Every feature for a given day uses only earlier days, which prevents leakage.\n"
                "- Multi-day forecasts are recursive: each prediction becomes history for the next day.\n"
                "- Accuracy is measured with a rolling-origin backtest against two naive baselines.\n"
                "- Metrics: MAE, RMSE and WAPE (total absolute error / total actual demand).")
    with right:
        with ui.card():
            ui.section_header("Inventory methodology")
            st.markdown(
                "- **Reorder point** = expected daily demand x lead time + safety stock.\n"
                "- **Safety stock** = z x standard deviation of daily demand x sqrt(lead time).\n"
                "- **Status:** Critical at or below the reorder point; Warning up to 1.5x; otherwise Healthy.\n"
                "- **Order quantity** covers lead time plus 14 days of demand plus safety stock, less current stock.\n"
                "- Expected demand uses the model forecast when available, otherwise the 28-day average.")
    with ui.card():
        ui.section_header("Known limitations")
        st.markdown(
            "- Synthetic data: no promotions, holidays, price changes or real stock-outs.\n"
            "- The generator restocks without supplier lead times, so stock levels do not follow the planning policy.\n"
            "- Quantities are treated as demand; real data would need stock-out flags.\n"
            "- Total Transactions counts daily product sales records, because receipt-level data is not available.")
    with ui.card():
        ui.section_header("Data preview", "First ten records")
        ui.table(df.head(10), [{"key": c, "label": c.replace("_", " "),
                                "align": "right" if c in ("Quantity", "Unit_Price", "Cost_Price", "Total_Sales", "Profit", "Stock_Remaining") else "left",
                                "fmt": (lambda v: v.strftime("%d %b %Y")) if c == "Date" else None}
                               for c in df.columns])
        ui.download("Download full dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                    "retail_sales.csv", key="dl_dataset")
