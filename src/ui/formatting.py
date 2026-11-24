"""Number and date formatting helpers (Pakistani Rupees). No UI dependencies."""
from __future__ import annotations

import math

DASH = "—"


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pkr(x, decimals: int = 0) -> str:
    """Full rupee amount, e.g. Rs. 1,234,567."""
    return DASH if _bad(x) else f"Rs. {x:,.{decimals}f}"


def pkr_compact(x) -> str:
    """Compact rupee amount for KPI cards, e.g. Rs. 12.4M."""
    if _bad(x):
        return DASH
    a = abs(x)
    if a >= 1e9:
        return f"Rs. {x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"Rs. {x / 1e6:,.2f}M"
    if a >= 1e4:
        return f"Rs. {x / 1e3:,.1f}K"
    return f"Rs. {x:,.0f}"


def num(x, decimals: int = 0) -> str:
    return DASH if _bad(x) else f"{x:,.{decimals}f}"


def pct(x, decimals: int = 1) -> str:
    """Format a fraction (0.123) as a percentage string."""
    return DASH if _bad(x) else f"{x * 100:.{decimals}f}%"


def pct_points(x, decimals: int = 1) -> str:
    """Format a value that is already in percent (12.3) with a % sign."""
    return DASH if _bad(x) else f"{x:.{decimals}f}%"


def date_label(d, long: bool = False) -> str:
    return d.strftime("%a, %d %b %Y") if long else d.strftime("%d %b %Y")
