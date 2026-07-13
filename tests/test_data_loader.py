"""Unit tests for ``app.data_loader``.

These tests don't need Streamlit to be installed — they exercise the pure
data-shaping helpers used by the dashboard. Run with:

    python -m pytest tests/

(``pytest`` is not in ``space/requirements.txt`` because the dashboard is
read-only on HF Spaces; install pytest locally when iterating on the tests.)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make ``app.*`` importable without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import data_loader as dl  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_equity() -> pd.Series:
    """20 trading days, monotonically rising 1% per day → clean numbers."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    equity = pd.Series([1.0 * (1.01 ** i) for i in range(20)], index=dates)
    return equity


@pytest.fixture
def tmp_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect REPORT_DIR to a tmp folder so we can write synthetic JSON."""
    monkeypatch.setattr(dl, "REPORT_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_daily_returns_no_lookahead(fake_equity: pd.Series) -> None:
    rets = dl.daily_returns(fake_equity)
    assert len(rets) == len(fake_equity)
    # Day 0 has no prior day → must be 0 (no NaN propagated to charts)
    assert rets.iloc[0] == 0.0
    # Day 1 should be exactly 0.01 (1% synthetic up-day)
    assert rets.iloc[1] == pytest.approx(0.01)


def test_rolling_sharpe_on_synthetic_uptrend(fake_equity: pd.Series) -> None:
    rets = dl.daily_returns(fake_equity)
    rs = dl.rolling_sharpe(rets, window=5, periods_per_year=252)
    # On a perfectly monotonic uptrend the rolling Sharpe blows up (std → 0)
    # so we expect inf or NaN once std hits 0 — either is acceptable.
    assert (rs.dropna().abs() > 0).any()


def test_drawdown_curve_matches_known_max(fake_equity: pd.Series) -> None:
    dd = dl.drawdown_curve(fake_equity)
    # Monotonic uptrend → drawdown must be 0 everywhere
    assert dd["drawdown"].max() == 0.0
    assert dd["drawdown"].min() == 0.0


def test_drawdown_curve_with_peak_and_trough() -> None:
    # Peak at 2.0, trough at 1.5 → -25% drawdown
    eq = pd.Series([1.0, 2.0, 1.5, 2.5], index=pd.bdate_range("2024-01-01", periods=4))
    dd = dl.drawdown_curve(eq)["drawdown"]
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0
    assert dd.iloc[2] == pytest.approx(-0.25)
    assert dd.iloc[3] == 0.0  # new high


def test_monthly_returns_shape_and_known_value() -> None:
    # Two calendar years so the pivot has more than one row.
    dates = pd.bdate_range("2023-11-01", periods=250)
    vals = []
    for i in range(250):
        vals.append(1.0 * (1.005 ** min(i, 60)))
    eq = pd.Series(vals, index=dates)
    matrix = dl.monthly_returns(eq)
    # First two calendar years must be present as rows.
    assert matrix.shape[0] >= 2
    assert 2023 in matrix.index
    assert 2024 in matrix.index
    # Months without data should be NaN, not 0 (the heatmap relies on this to
    # show grey "no data" rather than a coloured "0%" cell).
    for y in matrix.index:
        assert matrix.loc[y].isna().sum() > 0


def test_underwater_intensity_columns_and_years() -> None:
    dates = pd.bdate_range("2023-06-01", periods=252)
    vals = [1.0 * (1.005 ** i) if i < 100 else 0.7 + (i - 100) * 0.001 for i in range(252)]
    eq = pd.Series(vals, index=dates)
    uw = dl.underwater_intensity(eq)
    assert list(uw.columns) == ["year", "doy", "drawdown"]
    assert sorted(uw["year"].unique().tolist()) == [2023, 2024]
    assert uw["drawdown"].min() < 0  # we engineered a trough


def test_fmt_pct_and_fmt_num_handle_nan() -> None:
    assert dl.fmt_pct(0.123) == "12.30%"
    assert dl.fmt_pct(float("nan")) == "—"
    assert dl.fmt_pct(None) == "—"
    assert dl.fmt_num(1.23456) == "1.235"
    assert dl.fmt_num(float("inf")) == "—"


def test_annual_return_on_flat_equity() -> None:
    eq = pd.Series([1.0] * 252, index=pd.bdate_range("2024-01-01", periods=252))
    assert dl.annual_return(eq) == pytest.approx(0.0, abs=1e-9)


def test_annual_return_returns_total_return_for_short_windows() -> None:
    # Two points 1 business day apart → CAGR would be numerically unstable
    # (the helper falls back to total return for windows shorter than 30
    # days).
    eq = pd.Series([1.0, 1.5], index=pd.bdate_range("2020-01-01", periods=2))
    assert dl.annual_return(eq) == pytest.approx(0.5)


def test_annual_return_compounds_over_long_window() -> None:
    # Five years of data → expect CAGR ≈ (1.5)^(1/5) - 1 ≈ 8.45%.
    idx = pd.DatetimeIndex(["2020-01-01", "2025-01-01"])
    eq = pd.Series([1.0, 1.5], index=idx)
    ar = dl.annual_return(eq)
    assert 0.08 < ar < 0.09


def test_subperiods_frame_handles_missing_block() -> None:
    out = dl.subperiods_frame(None)
    assert out.empty
    out = dl.subperiods_frame([])
    assert out.empty


# ---------------------------------------------------------------------------
# JSON loaders (with redirected REPORT_DIR)
# ---------------------------------------------------------------------------


def test_load_portfolio_metrics_returns_dict(tmp_reports: Path) -> None:
    (tmp_reports / "portfolio_metrics.json").write_text(
        json.dumps({"sharpe": 0.86, "cagr": 0.215}), encoding="utf-8"
    )
    out = dl.load_portfolio_metrics()
    assert out["sharpe"] == 0.86
    assert out["cagr"] == 0.215


def test_load_missing_file_returns_default(tmp_reports: Path) -> None:
    assert dl.load_portfolio_metrics() == {}
    assert dl.load_buy_hold_metrics() == {}
    assert dl.load_subperiods() == {}
    assert dl.load_significance() == {}


def test_load_subperiods_parses_dates(tmp_reports: Path) -> None:
    payload = {
        "yearly": [
            {
                "period": "FY2024",
                "start": "2024-01-01",
                "end": "2024-06-30",
                "sharpe": 1.2,
                "n_days": 126,
            }
        ],
        "folds": [],
        "regimes": [],
    }
    (tmp_reports / "robustness_subperiods.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    out = dl.load_subperiods()
    df = dl.subperiods_frame(out["yearly"])
    assert len(df) == 1
    assert df["start"].iloc[0] == pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# Live data sanity check (only runs if the user has the real reports/)
# ---------------------------------------------------------------------------


def test_real_portfolio_metrics_sane() -> None:
    """If reports/ exists, sanity-check the published numbers haven't drifted
    in a way that would crash the dashboard. This is a smoke test, not a
    snapshot.
    """
    pm = dl.load_portfolio_metrics()
    if not pm:
        pytest.skip("No reports/ in repo; skipping live sanity check.")
    assert -1.0 < pm.get("sharpe", 0.0) < 5.0
    assert -1.0 < pm.get("cagr", 0.0) < 5.0
    assert pm.get("max_drawdown", 0.0) <= 0.0
    assert pm.get("n_days", 0) > 100