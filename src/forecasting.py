"""Demand forecasting: feature engineering, baselines, backtesting and forecasting.

Design notes
------------
* A single Random Forest is trained on all products, using calendar features,
  lagged demand and rolling means (all computed from *past* data only).
* Multi-step forecasts are produced recursively: each day's prediction is fed
  back as history for the next day, so training and inference use the same
  feature definitions.
* Models are judged by a rolling-origin backtest against two naive baselines.
  A model that cannot beat them is not worth deploying.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LAGS = (1, 7, 14)
ROLL_WINDOWS = (7, 28)
CALENDAR = ["day_of_week", "month", "day_of_year", "is_weekend"]
FEATURES = (["product_code"] + CALENDAR
            + [f"lag_{k}" for k in LAGS]
            + [f"roll_mean_{w}" for w in ROLL_WINDOWS])
MIN_TRAIN_DAYS = 120


def demand_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sales into a (date x product) matrix of daily units, gaps filled with 0."""
    wide = (df.pivot_table(index="Date", columns="Product_ID",
                           values="Quantity", aggfunc="sum")
            .sort_index())
    full_range = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full_range).fillna(0.0)


def _product_code(product_id: str) -> int:
    return int(str(product_id).lstrip("P"))


def _calendar(date: pd.Timestamp) -> dict:
    return {"day_of_week": date.dayofweek, "month": date.month,
            "day_of_year": date.dayofyear, "is_weekend": int(date.dayofweek >= 5)}


def build_training_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Long-format training rows. Every feature at date t uses data before t only."""
    parts = []
    for pid in wide.columns:
        s = wide[pid]
        frame = pd.DataFrame({"Date": wide.index, "Product_ID": pid, "target": s.values})
        frame["product_code"] = _product_code(pid)
        frame["day_of_week"] = wide.index.dayofweek
        frame["month"] = wide.index.month
        frame["day_of_year"] = wide.index.dayofyear
        frame["is_weekend"] = (wide.index.dayofweek >= 5).astype(int)
        for k in LAGS:
            frame[f"lag_{k}"] = s.shift(k).values
        for w in ROLL_WINDOWS:
            frame[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean().values
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).dropna()


def fit_model(wide: pd.DataFrame) -> RandomForestRegressor:
    train = build_training_frame(wide)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["target"])
    return model


def _features_for_next_day(history: pd.DataFrame) -> pd.DataFrame:
    """Feature rows (one per product) for the day after the end of `history`."""
    date = history.index[-1] + pd.Timedelta(days=1)
    rows = {"product_code": [_product_code(p) for p in history.columns]}
    for key, value in _calendar(date).items():
        rows[key] = [value] * history.shape[1]
    for k in LAGS:
        rows[f"lag_{k}"] = history.iloc[-k].values
    for w in ROLL_WINDOWS:
        rows[f"roll_mean_{w}"] = history.tail(w).mean().values
    return pd.DataFrame(rows, index=history.columns)[FEATURES]


def forecast_model(model: RandomForestRegressor, wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Recursive multi-step forecast. Returns a (horizon x product) matrix."""
    history = wide.copy()
    for _ in range(horizon):
        preds = np.clip(model.predict(_features_for_next_day(history)), 0, None)
        next_date = history.index[-1] + pd.Timedelta(days=1)
        history.loc[next_date] = preds
    return history.iloc[-horizon:]


def forecast_seasonal_naive(wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Baseline: demand equals the value from the same weekday last week."""
    dates = pd.date_range(wide.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    last_week = wide.tail(7).values
    return pd.DataFrame([last_week[i % 7] for i in range(horizon)],
                        index=dates, columns=wide.columns)


def forecast_weekday_mean(wide: pd.DataFrame, horizon: int, weeks: int = 8) -> pd.DataFrame:
    """Baseline: average demand per weekday over the last `weeks` weeks."""
    recent = wide.tail(7 * weeks)
    by_dow = recent.groupby(recent.index.dayofweek).mean()
    dates = pd.date_range(wide.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame(by_dow.loc[dates.dayofweek].values, index=dates, columns=wide.columns)


def _score(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {"MAE": float(mean_absolute_error(actual, pred)),
            "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
            "WAPE": float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))}


def backtest(df: pd.DataFrame, horizon: int = 14, n_folds: int = 4) -> pd.DataFrame:
    """Rolling-origin evaluation: train on data up to each cutoff, forecast `horizon`
    days ahead, compare with what actually happened. Scores are pooled across folds."""
    wide = demand_matrix(df)
    collected = {"Random Forest": ([], []), "Weekday mean": ([], []), "Seasonal naive": ([], [])}
    for fold in range(n_folds, 0, -1):
        cut = len(wide) - fold * horizon
        if cut < MIN_TRAIN_DAYS:
            continue
        train, actual = wide.iloc[:cut], wide.iloc[cut:cut + horizon]
        forecasts = {
            "Random Forest": forecast_model(fit_model(train), train, horizon),
            "Weekday mean": forecast_weekday_mean(train, horizon),
            "Seasonal naive": forecast_seasonal_naive(train, horizon),
        }
        for name, pred in forecasts.items():
            collected[name][0].append(actual.values.ravel())
            collected[name][1].append(pred.values.ravel())
    if not collected["Random Forest"][0]:
        raise ValueError("Not enough history for backtesting.")
    rows = [{"Model": name, **_score(np.concatenate(a), np.concatenate(p))}
            for name, (a, p) in collected.items()]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def forecast_products(df: pd.DataFrame, horizon: int = 14):
    """Train on all history and forecast `horizon` days.

    Returns (daily forecast, per-product summary, backtest scores).
    """
    wide = demand_matrix(df)
    predicted = forecast_model(fit_model(wide), wide, horizon)
    names = df.drop_duplicates("Product_ID").set_index("Product_ID")["Product"]
    forecast = (predicted.rename_axis("Date").reset_index()
                .melt(id_vars="Date", var_name="Product_ID", value_name="Predicted_Demand"))
    forecast["Product"] = forecast["Product_ID"].map(names)
    forecast = forecast[["Date", "Product_ID", "Product", "Predicted_Demand"]]
    summary = (forecast.groupby(["Product_ID", "Product"], as_index=False)["Predicted_Demand"]
               .sum().sort_values("Predicted_Demand", ascending=False).reset_index(drop=True))
    return forecast, summary, backtest(df, horizon)


def holdout_predictions(df: pd.DataFrame, horizon: int = 14):
    """Train on all but the last `horizon` days and forecast them (for actual-vs-predicted charts).

    Uses the same fit_model / forecast_model as the production forecast.
    Returns (actual, predicted), each a (horizon x product) matrix.
    """
    wide = demand_matrix(df)
    if len(wide) < MIN_TRAIN_DAYS + horizon:
        raise ValueError("Not enough history for a holdout evaluation.")
    train, actual = wide.iloc[:-horizon], wide.iloc[-horizon:]
    return actual, forecast_model(fit_model(train), train, horizon)
