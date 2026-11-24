"""Retail Intelligence System: Streamlit entry point (navigation, filters, page routing)."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Retail Intelligence System", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from ui import components, services, views  # noqa: E402  (after sys.path setup)
from ui.formatting import date_label  # noqa: E402

components.load_css()
st.logo(str(ROOT / "src/ui/assets/logo.svg"), icon_image=str(ROOT / "src/ui/assets/icon.svg"), size="large")

try:
    df = services.get_data()
except FileNotFoundError:
    st.error("Dataset not found. Run `python src/generate_data.py` from the project folder, then reload.")
    st.stop()
except ValueError as exc:
    st.error(f"The dataset could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown('<div class="ri-side-label">Filters</div>', unsafe_allow_html=True)
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    picked = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                           help="Applies to Executive Overview, Sales Analytics and Product Intelligence.")
    category = st.selectbox("Category", ["All"] + sorted(df["Category"].unique().tolist()),
                            format_func=lambda c: "All categories" if c == "All" else c,
                            help="Applies to every page except Data and Methodology.")
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    else:
        start, end = pd.Timestamp(min_d), pd.Timestamp(max_d)
        st.caption("Select an end date to apply the range.")
    st.markdown('<div class="ri-side-foot">Portfolio prototype using synthetic retail data.<br>'
                f'Data: {date_label(df["Date"].min())} – {date_label(df["Date"].max())}</div>',
                unsafe_allow_html=True)

view = df[df["Date"].between(start, end)]
if category != "All":
    view = view[view["Category"] == category]
ctx = views.FilterContext(df=df, view=view, start=start, end=end, category=category)


def overview_page():
    views.executive_overview(ctx)


def sales_page():
    views.sales_analytics(ctx)


def product_page():
    views.product_intelligence(ctx)


def inventory_page():
    views.inventory_planning(ctx)


def forecast_page():
    views.demand_forecasting(ctx)


def methodology_page():
    views.data_and_methodology(ctx)


navigation = st.navigation([
    st.Page(overview_page, title="Executive Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(sales_page, title="Sales Analytics", icon=":material/monitoring:", url_path="sales"),
    st.Page(product_page, title="Product Intelligence", icon=":material/category:", url_path="products"),
    st.Page(inventory_page, title="Inventory Planning", icon=":material/inventory_2:", url_path="inventory"),
    st.Page(forecast_page, title="Demand Forecasting", icon=":material/query_stats:", url_path="forecast"),
    st.Page(methodology_page, title="Data and Methodology", icon=":material/menu_book:", url_path="methodology"),
])
navigation.run()
