"""Cached data and model services (the only place that wraps work in st.cache_data)."""
import streamlit as st

from analytics import load_data
from forecasting import forecast_products, holdout_predictions


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_data(show_spinner=False)
def run_forecast(horizon: int):
    return forecast_products(get_data(), horizon=horizon)


@st.cache_data(show_spinner=False)
def run_holdout(horizon: int):
    return holdout_predictions(get_data(), horizon=horizon)
