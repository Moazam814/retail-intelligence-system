"""Generate the synthetic retail dataset (deterministic for a given seed)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# (id, name, category, unit price, cost price, base daily demand)
PRODUCTS = [
    ("P001", "Milk", "Dairy", 220, 180, 8.0),
    ("P002", "Bread", "Bakery", 100, 70, 10.0),
    ("P003", "Rice", "Grocery", 350, 280, 5.0),
    ("P004", "Cooking Oil", "Grocery", 550, 470, 4.0),
    ("P005", "Eggs", "Dairy", 300, 250, 6.0),
    ("P006", "Tea", "Beverages", 450, 370, 3.5),
    ("P007", "Sugar", "Grocery", 160, 130, 5.5),
    ("P008", "Biscuits", "Snacks", 120, 85, 7.0),
    ("P009", "Juice", "Beverages", 180, 130, 4.5),
    ("P010", "Soap", "Household", 140, 100, 3.0),
    ("P011", "Shampoo", "Personal Care", 350, 270, 2.5),
    ("P012", "Detergent", "Household", 280, 220, 3.0),
]


def generate(seed: int = 42, start: str = "2025-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Daily sales with weekly + annual seasonality, Poisson noise and stock replenishment."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for pid, name, category, price, cost, base_demand in PRODUCTS:
        stock = int(rng.integers(90, 150))
        for date in dates:
            weekend = 1.25 if date.dayofweek >= 5 else 1.0
            seasonal = 1 + 0.18 * np.sin(2 * np.pi * date.dayofyear / 365)
            monthly_factor = 1.12 if date.month in [6, 7, 8, 12] else 1.0
            demand = int(rng.poisson(base_demand * weekend * seasonal * monthly_factor))
            if stock < 25:
                stock += int(rng.integers(70, 130))
            sold = min(demand, stock)
            stock -= sold
            rows.append({"Date": date, "Product_ID": pid, "Product": name,
                         "Category": category, "Quantity": sold,
                         "Unit_Price": price, "Cost_Price": cost,
                         "Total_Sales": sold * price, "Profit": sold * (price - cost),
                         "Stock_Remaining": stock})
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_DIR / "retail_sales.csv", index=False)
    print(f"Created {len(df):,} synthetic daily product records at data/retail_sales.csv")


if __name__ == "__main__":
    main()
