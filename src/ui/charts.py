"""Plotly figure builders with one consistent light, blue-accent theme."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

NAVY, BLUE, MID_BLUE = "#0F2A4A", "#2563EB", "#60A5FA"
LIGHT_BLUE, SKY = "#BFDBFE", "#DBEAFE"
SLATE, GRID = "#64748B", "#E8EDF5"
GREEN, AMBER, RED = "#16A34A", "#D97706", "#DC2626"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Arial, sans-serif"
RULES = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}


def _style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=4, r=4, t=28, b=4),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family=FONT, size=12, color=NAVY),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=NAVY)))
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID, title_text="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat="~s", title_text="")
    return fig


def aggregate(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Revenue / profit / units totals per period (rule: 'D', 'W' or 'MS')."""
    out = (df.set_index("Date")[["Total_Sales", "Profit", "Quantity"]].resample(rule).sum()
           .rename(columns={"Total_Sales": "Revenue", "Quantity": "Units"}))
    out["Margin"] = (out["Profit"] / out["Revenue"]).where(out["Revenue"] > 0)
    return out.reset_index()


def revenue_profit_trend(df: pd.DataFrame, rule: str = "W") -> go.Figure:
    t = aggregate(df, rule)
    mode = "lines+markers" if len(t) <= 40 else "lines"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["Date"], y=t["Revenue"], name="Revenue", mode=mode,
                             line=dict(color=BLUE, width=2.5), fill="tozeroy",
                             fillcolor="rgba(37,99,235,0.08)", hovertemplate="Rs. %{y:,.0f}"))
    fig.add_trace(go.Scatter(x=t["Date"], y=t["Profit"], name="Profit", mode=mode,
                             line=dict(color=NAVY, width=2), hovertemplate="Rs. %{y:,.0f}"))
    return _style(fig)


def category_revenue_bar(cats: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=cats["Revenue"], y=cats["Category"], orientation="h", marker_color=BLUE,
        text=[f"Rs. {v / 1e3:,.0f}K" for v in cats["Revenue"]], textposition="outside",
        cliponaxis=False, hovertemplate="Rs. %{x:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickformat="")
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(hovermode="closest")
    return _style(fig, height=340, legend=False)


def top_products_bar(products: pd.DataFrame, n: int = 5) -> go.Figure:
    top = products.head(n)
    fig = go.Figure(go.Bar(
        x=top["Revenue"], y=top["Product"], orientation="h", marker_color=BLUE,
        text=[f"Rs. {v / 1e3:,.0f}K" for v in top["Revenue"]], textposition="outside",
        cliponaxis=False, hovertemplate="Rs. %{x:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickformat="")
    fig.update_xaxes(showticklabels=False)
    return _style(fig, height=300, legend=False).update_layout(hovermode="closest")


def monthly_revenue_bar(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=monthly["Month"], y=monthly["Revenue"], marker_color=BLUE, name="Revenue",
                           hovertemplate="Rs. %{y:,.0f}"))
    return _style(fig, legend=False)


def profit_margin_trend(df: pd.DataFrame, rule: str = "W") -> go.Figure:
    t = aggregate(df, rule)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["Date"], y=t["Profit"], name="Profit", marker_color=LIGHT_BLUE,
                         hovertemplate="Rs. %{y:,.0f}"))
    fig.add_trace(go.Scatter(x=t["Date"], y=t["Margin"], name="Profit margin", yaxis="y2",
                             mode="lines", line=dict(color=NAVY, width=2), hovertemplate="%{y:.1%}"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, tickformat=".0%",
                                  rangemode="tozero"))
    return _style(fig)


def category_comparison(cats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats["Category"], y=cats["Revenue"], name="Revenue", marker_color=BLUE,
                         hovertemplate="Rs. %{y:,.0f}"))
    fig.add_trace(go.Bar(x=cats["Category"], y=cats["Profit"], name="Profit", marker_color=LIGHT_BLUE,
                         hovertemplate="Rs. %{y:,.0f}"))
    fig.update_layout(barmode="group", bargap=0.25)
    return _style(fig)


def daily_revenue_histogram(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("Date")["Total_Sales"].sum()
    fig = go.Figure(go.Histogram(x=daily, nbinsx=20, marker_color=BLUE, marker_line_color="white",
                                 marker_line_width=1, hovertemplate="Rs. %{x:,.0f}<br>%{y} days<extra></extra>"))
    fig.add_shape(type="line", x0=float(daily.median()), x1=float(daily.median()), y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color=NAVY, width=1.5, dash="dash"))
    fig.add_annotation(x=float(daily.median()), y=1, yref="paper", text="Median day", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color=SLATE))
    fig.update_xaxes(tickformat="~s")
    fig.update_yaxes(title_text="Days", tickformat="")
    fig.update_layout(hovermode="closest", bargap=0.04)
    return _style(fig, legend=False)


def weekday_pattern(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("Date")["Total_Sales"].sum()
    by_dow = daily.groupby(daily.index.dayofweek).mean().reindex(range(7))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = [BLUE if i < 5 else NAVY for i in range(7)]
    fig = go.Figure(go.Bar(x=names, y=by_dow.values, marker_color=colors,
                           hovertemplate="Rs. %{y:,.0f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _style(fig, legend=False)


def margin_scatter(products: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=products["Revenue"], y=products["Margin_%"], mode="markers+text", text=products["Product"],
        textposition="top center", marker=dict(size=12, color=BLUE, line=dict(color="white", width=1.5)),
        hovertemplate="%{text}<br>Revenue: Rs. %{x:,.0f}<br>Margin: %{y:.1f}%<extra></extra>"))
    fig.update_xaxes(title_text="Revenue (PKR)", tickformat="~s")
    fig.update_yaxes(title_text="Profit margin (%)", tickformat=".0f")
    fig.update_layout(hovermode="closest")
    return _style(fig, height=360, legend=False)


STATUS_COLORS = {"Critical": RED, "Warning": AMBER, "Healthy": GREEN}


def inventory_chart(inv: pd.DataFrame) -> go.Figure:
    """inv needs: Product, Stock_Remaining, Reorder_Point, Level."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=inv["Product"], y=inv["Stock_Remaining"], name="Current stock",
                         marker_color=[STATUS_COLORS.get(s, BLUE) for s in inv["Level"]],
                         hovertemplate="%{y:,.0f} units"))
    fig.add_trace(go.Scatter(x=inv["Product"], y=inv["Reorder_Point"], name="Reorder point", mode="markers",
                             marker=dict(symbol="diamond", size=10, color=NAVY, line=dict(color="white", width=1)),
                             hovertemplate="%{y:,.0f} units"))
    fig.update_yaxes(tickformat=",", title_text="Units")
    return _style(fig)


def model_comparison(scores: pd.DataFrame) -> go.Figure:
    colors = [BLUE if m == "Random Forest" else LIGHT_BLUE for m in scores["Model"]]

    fig = go.Figure(go.Bar(
        x=scores["Model"],
        y=scores["MAE"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in scores["MAE"]],
        textposition="outside",
        hovertemplate="MAE %{y:.2f} units<extra></extra>"
    ))

    fig.update_layout(hovermode="closest")

    fig = _style(fig, height=300, legend=False)

    fig.update_yaxes(
        title_text="MAE (units/day, lower is better)",
        tickformat=".1f",
        rangemode="tozero"
    )

    return fig


def forecast_chart(history: pd.Series, backtest_pred: pd.Series, future: pd.Series) -> go.Figure:
    """Recent actuals, the model's holdout predictions for the last days, and the future forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Actual demand", mode="lines+markers",
                             line=dict(color=NAVY, width=2), marker=dict(size=4),
                             hovertemplate="%{y:.0f} units"))
    fig.add_trace(go.Scatter(x=backtest_pred.index, y=backtest_pred.values, name="Predicted (holdout)",
                             mode="lines+markers", line=dict(color=MID_BLUE, width=2, dash="dot"),
                             marker=dict(size=4), hovertemplate="%{y:.1f} units"))
    fig.add_trace(go.Scatter(x=future.index, y=future.values, name="Forecast (next 14 days)",
                             mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=5),
                             hovertemplate="%{y:.1f} units"))
    if len(history):
        fig.add_shape(type="line", x0=str(history.index.max()), x1=str(history.index.max()),
                      y0=0, y1=1, xref="x", yref="paper", line=dict(color=SLATE, width=1, dash="dash"))
    fig.update_yaxes(title_text="Units per day", tickformat=",", rangemode="tozero")
    return _style(fig, height=380)
