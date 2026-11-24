import pytest
from analytics import kpis, monthly_performance, inventory_snapshot, load_data
from helpers import make_df


def test_kpis_match_manual_totals():
    df = make_df(days=10, products=2)
    k = kpis(df)
    assert k["Revenue"] == df["Total_Sales"].sum()
    assert k["Products"] == 2
    assert k["Units sold"] == df["Quantity"].sum()


def test_monthly_performance_has_one_row_per_month():
    assert len(monthly_performance(make_df(days=90))) == 3


def test_load_data_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Date,Product_ID\n2025-01-01,P001\n")
    with pytest.raises(ValueError):
        load_data(bad)


def test_inventory_flags_low_stock_and_healthy_stock():
    df = make_df(days=60, products=2)
    latest = df["Date"].max()
    df.loc[(df["Date"] == latest) & (df["Product_ID"] == "P001"), "Stock_Remaining"] = 3
    df.loc[(df["Date"] == latest) & (df["Product_ID"] == "P002"), "Stock_Remaining"] = 5000
    snap = inventory_snapshot(df).set_index("Product_ID")
    assert snap.loc["P001", "Status"] == "Reorder"
    assert snap.loc["P002", "Status"] == "Healthy"
    assert snap.loc["P001", "Suggested_Order_Qty"] > 0
    assert snap.loc["P002", "Suggested_Order_Qty"] == 0
