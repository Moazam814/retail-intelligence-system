import numpy as np
import pandas as pd
from forecasting import (FEATURES, backtest, build_training_frame, demand_matrix,
                         forecast_products, forecast_seasonal_naive)
from helpers import make_df


def test_features_use_only_past_data():
    wide = demand_matrix(make_df(days=60, products=1))
    wide.iloc[:, 0] = np.arange(len(wide), dtype=float)       # value == day index
    frame = build_training_frame(wide).set_index("Date")
    day = wide.index[40]
    assert frame.loc[day, "lag_1"] == 39
    assert frame.loc[day, "lag_7"] == 33
    assert frame.loc[day, "roll_mean_7"] == np.mean(np.arange(33, 40))   # excludes today


def test_seasonal_naive_is_exact_on_weekly_pattern():
    wide = demand_matrix(make_df(days=70))
    pred = forecast_seasonal_naive(wide.iloc[:-14], 14)
    assert np.allclose(pred.values, wide.iloc[-14:].values)


def test_forecast_shape_dates_and_non_negative():
    df = make_df(days=150, products=3)
    forecast, summary, scores = forecast_products(df, horizon=14)
    assert len(forecast) == 14 * 3
    assert forecast["Date"].min() == df["Date"].max() + pd.Timedelta(days=1)
    assert (forecast["Predicted_Demand"] >= 0).all()
    assert len(summary) == 3


def test_backtest_reports_all_models_with_finite_scores():
    scores = backtest(make_df(days=200), horizon=14, n_folds=2)
    assert set(scores["Model"]) == {"Random Forest", "Weekday mean", "Seasonal naive"}
    assert np.isfinite(scores[["MAE", "RMSE", "WAPE"]].values).all()
    assert list(FEATURES)[0] == "product_code"
