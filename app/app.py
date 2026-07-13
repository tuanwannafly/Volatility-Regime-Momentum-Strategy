"""Interactive Streamlit dashboard for the volatility-regime momentum project.

Run locally:

    streamlit run app/app.py

Deploy to Hugging Face Spaces (Streamlit SDK) by pushing ``app/`` plus the
``space/`` artifacts to a Space repo. See README_DASHBOARD.md for the full
walkthrough.

The dashboard is read-only with respect to ``reports/`` -- nothing here
re-runs training or backtests. If those artifacts are out of date, regenerate
them upstream (``python -m src.backtest.run`` then ``python -m src.robustness``)
and refresh the page.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# Make ``app.*`` imports work whether Streamlit runs from the repo root or from
# inside the ``app/`` folder.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
    HAS_ALTAIR = True
except Exception:  # pragma: no cover -- exercised only when altair is missing
    HAS_ALTAIR = False

from cache import cache
from data_loader import (
    annual_return,
    buy_hold_curve,
    daily_returns,
    drawdown_curve,
    fmt_num,
    fmt_pct,
    load_buy_hold_metrics,
    load_equity_curve,
    load_portfolio_metrics,
    load_significance,
    load_subperiods,
    monthly_returns,
    rolling_sharpe,
    subperiods_frame,
    underwater_intensity,
)


# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VN30 Regime Momentum — Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A muted, terminal-ish palette so the dashboard reads as research software
# rather than a marketing page. Defined once so every chart pulls from the
# same dictionary.
PALETTE = {
    "strategy": "#2563eb",   # blue
    "buy_hold": "#f97316",   # orange
    "drawdown": "#dc2626",   # red
    "sharpe": "#0f766e",     # teal
    "ci": "#9ca3af",         # gray
    "grid": "#e5e7eb",       # light gray
    "text": "#111827",
}


@cache
def _style_overrides() -> str:
    return """
    <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 1rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        [data-testid="stMetricDelta"] { font-size: 0.85rem; }
        .stAlert p { margin: 0; }
        h1, h2, h3 { font-weight: 600; }
    </style>
    """


st.markdown(_style_overrides(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders cached on disk content
# ---------------------------------------------------------------------------


@cache
def _load_equity() -> pd.DataFrame:
    return load_equity_curve()


@cache
def _load_buy_hold_curve() -> pd.DataFrame | None:
    return buy_hold_curve()


@cache
def _load_portfolio_metrics() -> dict:
    return load_portfolio_metrics()


@cache
def _load_buy_hold_metrics() -> dict:
    return load_buy_hold_metrics()


@cache
def _load_subperiods() -> dict:
    return load_subperiods()


@cache
def _load_significance() -> dict:
    return load_significance()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    equity = _load_equity()
    if equity.empty:
        st.sidebar.warning("No equity curve found at reports/portfolio_equity_curve.csv.")
        default_start = pd.Timestamp("2021-06-16")
        default_end = pd.Timestamp("2025-12-16")
    else:
        default_start = pd.Timestamp(equity["time"].min())
        default_end = pd.Timestamp(equity["time"].max())

    st.sidebar.title("Controls")
    st.sidebar.caption("All filters below only reshape what's *shown*. "
                       "Metrics shown on the rest of the page are point estimates "
                       "of the *full* backtest.")

    st.sidebar.subheader("Date range")
    # Slider for a date range is more mobile-friendly than two stacked
    # date pickers, and Streamlit 1.36+ supports datetime directly.
    date_window = st.sidebar.slider(
        "Window",
        min_value=default_start.to_pydatetime(),
        max_value=default_end.to_pydatetime(),
        value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
        format="YYYY-MM-DD",
        key="date_range_slider",
        help="Drag to zoom in on a sub-period. Equity curve and rolling stats follow.",
    )
    if isinstance(date_window, tuple) and len(date_window) == 2:
        start_date, end_date = date_window
    else:
        start_date, end_date = default_start, default_end

    st.sidebar.subheader("Rolling analytics")
    rolling_window = st.sidebar.slider(
        "Rolling Sharpe window (trading days)",
        min_value=21,
        max_value=252,
        value=63,
        step=21,
        help="63 ≈ 3 months, 126 ≈ 6 months, 252 ≈ 1 year.",
    )
    vol_window = st.sidebar.slider(
        "Rolling volatility window (days)",
        min_value=21,
        max_value=252,
        value=63,
        step=21,
    )

    return pd.Timestamp(start_date), pd.Timestamp(end_date), int(rolling_window), int(vol_window)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def render_header() -> None:
    portfolio = _load_portfolio_metrics()
    buy_hold = _load_buy_hold_metrics()

    st.title("Volatility-Regime Momentum — VN30")
    st.markdown(
        "Interactive tearsheet over the walk-forward OOS backtest. "
        "All numbers come straight out of `reports/` — nothing here is "
        "recomputed at runtime."
    )

    cols = st.columns(6)
    metrics = [
        ("CAGR (portfolio)", fmt_pct(portfolio.get("cagr")) if portfolio else "—"),
        ("Sharpe (portfolio)", fmt_num(portfolio.get("sharpe")) if portfolio else "—"),
        ("Sortino", fmt_num(portfolio.get("sortino")) if portfolio else "—"),
        ("Max DD (portfolio)", fmt_pct(portfolio.get("max_drawdown")) if portfolio else "—"),
        ("Buy & hold CAGR", fmt_pct(buy_hold.get("avg", {}).get("cagr")) if buy_hold else "—"),
        ("Avg buy-hold Sharpe", fmt_num(buy_hold.get("avg", {}).get("sharpe")) if buy_hold else "—"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def slice_by_date(equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if equity.empty:
        return equity
    mask = (equity["time"] >= start) & (equity["time"] <= end)
    return equity.loc[mask].reset_index(drop=True)


def render_overview(
    start: pd.Timestamp,
    end: pd.Timestamp,
    rolling_window: int,
    vol_window: int,
) -> None:
    st.header("1 · Equity curve & drawdown")
    equity = _load_equity()
    if equity.empty:
        st.info("Run `python -m src.backtest.run` to produce portfolio_equity_curve.csv.")
        return

    window = slice_by_date(equity, start, end)
    if window.empty:
        st.warning("Selected date range contains no data — widen the window.")
        return

    equity_curve = window.set_index("time")["equity"]
    bh_curve = _load_buy_hold_curve()
    bh_window = None
    if bh_curve is not None and not bh_curve.empty:
        # Wide format: one column per ticker (normalised to 1.0 at first date)
        bh_indexed = bh_curve.set_index("time")
        bh_masked = bh_indexed.loc[(bh_indexed.index >= start) & (bh_indexed.index <= end)]
        if not bh_masked.empty:
            # Build an equal-weight benchmark by averaging per-day returns of
            # the available tickers, then normalising to the strategy's start.
            rets = bh_masked.pct_change().fillna(0.0)
            mean_ret = rets.mean(axis=1, skipna=True)
            eq = (1.0 + mean_ret).cumprod() * float(equity_curve.iloc[0])
            bh_window = pd.DataFrame({"buy_hold_eq_wt": eq})

    st.caption(
        f"Window: **{start.date()} → {end.date()}** — {len(window)} trading days "
        f"({'%.0f%%' % ((len(window) / max(len(equity), 1)) * 100)} of full sample)."
    )

    chart_df = pd.DataFrame({"Strategy": equity_curve})
    if bh_window is not None and not bh_window.empty:
        chart_df["Buy & hold (eq-wt)"] = bh_window["buy_hold_eq_wt"]

    st.line_chart(
        chart_df,
        color=[PALETTE["strategy"], PALETTE["buy_hold"]],
        height=320,
    )

    draw = drawdown_curve(equity_curve)
    st.area_chart(
        draw[["drawdown"]],
        color=[PALETTE["drawdown"]],
        height=160,
    )

    # Summary numbers for the *window* the user picked, so the headline tiles
    # below match what they see on the chart.
    rets = daily_returns(equity_curve)
    if not rets.empty and rets.std() and not math.isnan(rets.std()):
        window_metrics = {
            "Window return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
            "Window CAGR": annual_return(equity_curve),
            "Window vol": float(rets.std() * math.sqrt(252)),
            "Window Sharpe": float((rets.mean() / rets.std()) * math.sqrt(252)) if rets.std() else 0.0,
            "Window max DD": float(draw["drawdown"].min()),
        }
    else:
        window_metrics = {k: float("nan") for k in [
            "Window return", "Window CAGR", "Window vol", "Window Sharpe", "Window max DD",
        ]}

    cols = st.columns(5)
    keys = ["Window return", "Window CAGR", "Window vol", "Window Sharpe", "Window max DD"]
    formatter = {
        "Window return": lambda v: fmt_pct(v),
        "Window CAGR": lambda v: fmt_pct(v),
        "Window vol": lambda v: fmt_pct(v),
        "Window Sharpe": lambda v: fmt_num(v),
        "Window max DD": lambda v: fmt_pct(v),
    }
    for col, key in zip(cols, keys):
        col.metric(key, formatter[key](window_metrics[key]))


def render_rolling(
    start: pd.Timestamp,
    end: pd.Timestamp,
    rolling_window: int,
    vol_window: int,
) -> None:
    st.header("2 · Rolling diagnostics")
    equity = _load_equity()
    if equity.empty:
        return
    window = slice_by_date(equity, start, end)
    if window.empty:
        return

    eq = window.set_index("time")["equity"]
    rets = daily_returns(eq)
    rs = rolling_sharpe(rets, window=rolling_window).dropna()
    vol = (rets.rolling(vol_window).std() * math.sqrt(252)).dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Rolling Sharpe ({rolling_window}d)")
        st.line_chart(pd.DataFrame({"Sharpe": rs}), color=[PALETTE["sharpe"]], height=280)
        st.caption("Annualized. 0 line is overlaid for context.")
    with col2:
        st.subheader(f"Rolling vol ({vol_window}d, annualized)")
        st.line_chart(pd.DataFrame({"Vol": vol}), color=[PALETTE["buy_hold"]], height=280)


def render_exposure(start: pd.Timestamp, end: pd.Timestamp) -> None:
    st.header("3 · Exposure & turnover")
    equity = _load_equity()
    if equity.empty:
        return
    window = slice_by_date(equity, start, end)
    if window.empty:
        return

    exposed = window["exposure_on"].astype(int)
    exposure_pct = float(exposed.mean()) if len(exposed) else 0.0
    turnover_daily = (window["cost"].abs() / window["equity"]).fillna(0.0)
    annualized_turnover = float(turnover_daily.sum())  # costs already in equity units

    cols = st.columns(3)
    cols[0].metric("Time in market", fmt_pct(exposure_pct))
    cols[1].metric(
        "Cumulative turnover (cost / equity)",
        fmt_num(annualized_turnover),
        help="Sum of |cost| / equity across the selected window. A higher number means the strategy traded more aggressively.",
    )
    cols[2].metric(
        "Days with zero exposure",
        f"{(exposed == 0).sum()}",
    )

    st.line_chart(
        pd.DataFrame({"Exposure (1=on)": exposed.values}, index=window["time"]),
        color=[PALETTE["strategy"]],
        height=180,
    )


def render_subperiods() -> None:
    st.header("4 · Sub-period breakdown")
    sub = _load_subperiods()
    if not sub:
        st.info("Run `python -m src.robustness` to produce sub-period files.")
        return

    st.caption("Two views of the same data: by calendar year, and by walk-forward fold.")

    yearly = subperiods_frame(sub.get("yearly"))
    folds = subperiods_frame(sub.get("folds"))
    regimes = subperiods_frame(sub.get("regimes"))

    if not yearly.empty:
        st.subheader("By year")
        show_cols = ["period", "total_return", "sharpe", "annual_vol", "max_drawdown", "n_days"]
        present = [c for c in show_cols if c in yearly.columns]
        view = yearly[present].copy()
        for col in ["total_return", "annual_vol", "max_drawdown"]:
            if col in view.columns:
                view[col] = view[col].map(lambda v: fmt_pct(v))
        if "sharpe" in view.columns:
            view["sharpe"] = view["sharpe"].map(lambda v: fmt_num(v))
        if "n_days" in view.columns:
            view["n_days"] = view["n_days"].astype(int)
        st.dataframe(view, width="stretch", hide_index=True)
        # Bar chart for visual scan
        if "sharpe" in yearly.columns:
            st.bar_chart(
                yearly.set_index("period")[["sharpe"]],
                color=[PALETTE["sharpe"]],
                height=200,
            )

    if not folds.empty:
        st.subheader("By walk-forward fold")
        show_cols = ["period", "total_return", "sharpe", "max_drawdown", "n_days"]
        present = [c for c in show_cols if c in folds.columns]
        view = folds[present].copy()
        for col in ["total_return", "max_drawdown"]:
            if col in view.columns:
                view[col] = view[col].map(lambda v: fmt_pct(v))
        if "sharpe" in view.columns:
            view["sharpe"] = view["sharpe"].map(lambda v: fmt_num(v))
        if "n_days" in folds.columns:
            view["n_days"] = view["n_days"].astype(int)
        st.dataframe(view, width="stretch", hide_index=True)

    if not regimes.empty:
        st.subheader("By vol regime")
        show_cols = ["period", "total_return", "sharpe", "annual_vol", "max_drawdown", "n_days"]
        present = [c for c in show_cols if c in regimes.columns]
        view = regimes[present].copy()
        for col in ["total_return", "annual_vol", "max_drawdown"]:
            if col in view.columns:
                view[col] = view[col].map(lambda v: fmt_pct(v))
        if "sharpe" in view.columns:
            view["sharpe"] = view["sharpe"].map(lambda v: fmt_num(v))
        if "n_days" in regimes.columns:
            view["n_days"] = view["n_days"].astype(int)
        st.dataframe(view, width="stretch", hide_index=True)


def render_significance() -> None:
    st.header("5 · Statistical significance of the Sharpe")
    sig = _load_significance()
    if not sig:
        st.info("Run `python -m src.robustness` to compute the Sharpe significance block.")
        return

    naive = sig.get("naive_tstat", {})
    lo = sig.get("lo_tstat", {})
    boot = sig.get("block_bootstrap", {})
    dsr = sig.get("deflated_sharpe", {})

    if naive:
        st.metric(
            "Naive t-stat (treating days as IID)",
            f"{naive.get('t_stat', float('nan')):.2f}",
            help=f"n={naive.get('n_days', '—')}, p={naive.get('p_value_two_sided', float('nan')):.3f}",
        )
    if lo:
        st.metric(
            "Lo (2002) HAC t-stat",
            f"{lo.get('t_stat', float('nan')):.2f}",
            help=f"autocorr(1)={lo.get('autocorr_lag1', float('nan')):.3f}, "
                 f"Σ⁺ autocorr={lo.get('pos_autocorr_sum', float('nan')):.2f}, "
                 f"p={lo.get('p_value_two_sided', float('nan')):.3f}",
        )
    if boot:
        contains_zero = "contains 0" if boot.get("ci_contains_zero") else "excludes 0"
        st.metric(
            "Block-bootstrap 95% CI (block=21d)",
            f"[{boot.get('sr_ci_95_low', float('nan')):.2f}, "
            f"{boot.get('sr_ci_95_high', float('nan')):.2f}] — {contains_zero}",
            help=f"Fraction of bootstraps with SR ≤ 0: {boot.get('frac_bootstrap_negative', float('nan')):.3f}.",
        )
    if dsr:
        cons = dsr.get("min_trials_4", {})
        agg = dsr.get("max_trials_60", {})
        st.metric(
            "Deflated Sharpe — conservative (4 trials)",
            f"p={cons.get('dsr_p_value', float('nan')):.3f}",
            help=f"E[max SR under H0]={cons.get('expected_max_sr_under_null', float('nan')):.2f}",
        )
        st.metric(
            "Deflated Sharpe — aggressive (60 trials)",
            f"p={agg.get('dsr_p_value', float('nan')):.3f}",
            help=f"E[max SR under H0]={agg.get('expected_max_sr_under_null', float('nan')):.2f}",
        )

    interp = dsr.get("interpretation")
    if interp:
        st.caption(interp)


def render_drawdown_calendar() -> None:
    """Section 5b — calendar heatmap of monthly returns + underwater strip.

    Uses Altair for the heatmap (ships with Streamlit; pinned in
    ``space/requirements.txt``). If Altair is unavailable we degrade to plain
    dataframes so the page still renders.
    """
    st.header("5b · Drawdown calendar")
    equity_full = _load_equity()
    if equity_full.empty:
        return
    eq = equity_full.set_index("time")["equity"]

    cols = st.columns([1, 1])

    # ---- Monthly returns heatmap (year x month) ----
    with cols[0]:
        st.subheader("Monthly returns")
        monthly = monthly_returns(eq)
        if monthly.empty:
            st.info("Not enough data to build a monthly returns matrix.")
        else:
            st.caption("Cell = monthly total return. Grey = no data for that month.")
            if HAS_ALTAIR:
                long = (
                    monthly.stack(future_stack=True)
                    .rename("ret")
                    .reset_index()
                    .rename(columns={"level_1": "month"})
                )
                long["ret_pct"] = long["ret"] * 100.0
                month_labels = [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                ]
                long["month_label"] = long["month"].map(
                    lambda m: month_labels[int(m) - 1] if pd.notna(m) else ""
                )
                # Use a diverging red-white-blue palette; the 0-crossing is
                # fixed so the colour scale is comparable across years.
                max_abs = max(
                    1.0, float(np.nanmax(np.abs(long["ret_pct"].to_numpy())))
                ) if not long["ret_pct"].isna().all() else 1.0
                heatmap = (
                    alt.Chart(long.dropna(subset=["ret"]))
                    .mark_rect()
                    .encode(
                        x=alt.X(
                            "month_label:O",
                            sort=month_labels,
                            title=None,
                            axis=alt.Axis(labelAngle=0),
                        ),
                        y=alt.Y(
                            "year:O",
                            sort="descending",
                            title=None,
                        ),
                        color=alt.Color(
                            "ret_pct:Q",
                            scale=alt.Scale(
                                scheme="redblue",
                                domain=[-max_abs, max_abs],
                                reverse=True,
                            ),
                            legend=alt.Legend(title="Return %", format=".1f"),
                        ),
                        tooltip=[
                            "year:O",
                            "month_label:O",
                            alt.Tooltip("ret_pct:Q", format=".2f", title="Return %"),
                        ],
                    )
                    .properties(height=200)
                )
                text = (
                    alt.Chart(long.dropna(subset=["ret"]))
                    .mark_text(fontSize=10)
                    .encode(
                        x=alt.X("month_label:O", sort=month_labels),
                        y=alt.Y("year:O", sort="descending"),
                        text=alt.Text("ret_pct:Q", format=".1f"),
                        color=alt.condition(
                            alt.datum.ret_pct > 0,
                            alt.value("white"),
                            alt.value("white"),
                        ),
                    )
                )
                st.altair_chart(heatmap + text, width="stretch")
            else:
                # Fallback: coloured dataframe
                styled = monthly.style.format("{:.2%}", na_rep="—").background_gradient(
                    cmap="RdBu", axis=None, vmin=-0.1, vmax=0.1
                )
                st.dataframe(styled, width="stretch")

    # ---- Underwater intensity heatmap (year x day-of-year) ----
    with cols[1]:
        st.subheader("Underwater equity")
        uw = underwater_intensity(eq)
        if uw.empty:
            st.info("Not enough data to plot drawdowns.")
        else:
            st.caption(
                "Each row = a calendar year; intensity = drawdown depth on "
                "that trading day. White = at high-water mark."
            )
            if HAS_ALTAIR:
                chart = (
                    alt.Chart(uw)
                    .mark_rect()
                    .encode(
                        x=alt.X(
                            "doy:Q",
                            title="Day of year",
                            axis=alt.Axis(values=[1, 32, 60, 91, 121, 152, 182,
                                                  213, 244, 274, 305, 335]),
                        ),
                        y=alt.Y(
                            "year:O",
                            sort="descending",
                            title=None,
                        ),
                        color=alt.Color(
                            "drawdown:Q",
                            scale=alt.Scale(
                                scheme="reds",
                                domain=[-0.25, 0.0],
                            ),
                            legend=alt.Legend(
                                title="Drawdown",
                                format=".0%",
                            ),
                        ),
                        tooltip=[
                            "year:O",
                            "doy:Q",
                            alt.Tooltip("drawdown:Q", format=".2%", title="DD"),
                        ],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart, width="stretch")
            else:
                pivot = uw.pivot(index="year", columns="doy", values="drawdown")
                st.dataframe(
                    pivot.style.format("{:.2%}", na_rep="").background_gradient(
                        cmap="Reds", axis=None, vmin=-0.2, vmax=0.0
                    ),
                    width="stretch",
                )

    # ---- Worst drawdown episodes ----
    st.subheader("Top 5 drawdown episodes")
    dd = drawdown_curve(eq)["drawdown"]
    episodes: list[dict[str, Any]] = []
    in_dd = False
    start: pd.Timestamp | None = None
    trough_val = 0.0
    for ts, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = ts
            trough_val = val
        elif val < 0 and in_dd:
            if val < trough_val:
                trough_val = val
        elif val >= 0 and in_dd:
            end = ts
            episodes.append(
                {
                    "start": start,
                    "end": end,
                    "depth": trough_val,
                    "length_days": (end - start).days,
                }
            )
            in_dd = False
            start = None
            trough_val = 0.0
    # If we ended still in drawdown, treat the last observation as the end
    if in_dd and start is not None:
        end = dd.index[-1]
        episodes.append(
            {
                "start": start,
                "end": end,
                "depth": trough_val,
                "length_days": (end - start).days,
            }
        )
    if not episodes:
        st.caption("No drawdown episodes detected.")
    else:
        top = sorted(episodes, key=lambda e: e["depth"])[:5]
        ep_df = pd.DataFrame(top)
        ep_df["depth"] = ep_df["depth"].map(lambda v: fmt_pct(v))
        ep_df["length_days"] = ep_df["length_days"].astype(int)
        st.dataframe(ep_df, width="stretch", hide_index=True)


def render_per_ticker(selected_tickers: list[str] | None = None) -> None:
    st.header("6 · Strategy vs individual buy-and-hold")
    buy_hold = _load_buy_hold_metrics()
    portfolio = _load_portfolio_metrics()
    if not buy_hold or "per_ticker" not in buy_hold:
        st.info("No per-ticker buy-hold metrics available.")
        return

    rows = buy_hold["per_ticker"]
    df_all = pd.DataFrame(rows)
    if "ticker" not in df_all.columns:
        st.warning("Per-ticker block is missing the 'ticker' field.")
        return

    all_tickers = sorted(df_all["ticker"].astype(str).unique().tolist())
    default_selection = all_tickers if not selected_tickers else [
        t for t in selected_tickers if t in all_tickers
    ]
    selected = st.multiselect(
        "Tickers to compare",
        options=all_tickers,
        default=default_selection,
        help="Pick any subset of the VN30 universe. Charts and the table refresh in place.",
        key="ticker_multiselect",
    )
    if not selected:
        st.info("Pick at least one ticker to populate the charts.")
        return

    df = df_all[df_all["ticker"].isin(selected)].sort_values("sharpe", ascending=False)

    # Side-by-side scatter: strategy CAGR is a single number, so we plot it as
    # a horizontal line and compare each ticker.
    cols = st.columns(2)

    with cols[0]:
        st.subheader("Per-ticker Sharpe (buy & hold)")
        st.bar_chart(
            df.set_index("ticker")[["sharpe"]],
            color=[PALETTE["buy_hold"]],
            height=320,
            horizontal=True,
        )
    with cols[1]:
        st.subheader("Per-ticker CAGR (buy & hold)")
        if portfolio and "cagr" in portfolio:
            st.caption(f"Strategy CAGR = {fmt_pct(portfolio['cagr'])} "
                       f"(horizontal line on chart for reference)")
        # st.bar_chart horizontal still requires a Series-style frame
        df_chart = df.set_index("ticker")[["cagr"]]
        st.bar_chart(
            df_chart,
            color=[PALETTE["buy_hold"]],
            height=320,
            horizontal=True,
        )

    show_cols = ["ticker", "total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "n_days"]
    present = [c for c in show_cols if c in df.columns]
    view = df[present].copy()
    for col in ["total_return", "cagr", "max_drawdown", "win_rate"]:
        if col in view.columns:
            view[col] = view[col].map(lambda v: fmt_pct(v) if not pd.isna(v) else "—")
    if "sharpe" in view.columns:
        view["sharpe"] = view["sharpe"].map(lambda v: fmt_num(v) if not pd.isna(v) else "—")
    if "n_days" in view.columns:
        view["n_days"] = view["n_days"].astype(int)
    st.dataframe(view, width="stretch", hide_index=True)

    # Per-ticker equity overlay if buy-and-hold price parquets are available
    bh_curve = _load_buy_hold_curve()
    if bh_curve is not None and not bh_curve.empty:
        overlay_tickers = [t for t in selected if t in bh_curve.columns]
        if overlay_tickers:
            overlay = bh_curve[["time"] + overlay_tickers].copy()
            overlay = overlay.set_index("time")
            # Normalise each to 1.0 at first available date so they're comparable
            for t in overlay_tickers:
                first_valid = overlay[t].dropna().iloc[0] if overlay[t].notna().any() else 1.0
                overlay[t] = overlay[t] / float(first_valid) if first_valid else overlay[t]
            # Add strategy equity for direct comparison
            strategy_eq = _load_equity().set_index("time")["equity"]
            overlay["__strategy"] = strategy_eq.reindex(overlay.index, method="ffill")
            st.subheader("Per-ticker buy-and-hold overlay vs strategy")
            st.line_chart(overlay, height=320)
            st.caption(
                "Each ticker is normalised to 1.0 at its first available date. "
                "Strategy line is ffill'd to align to the buy-and-hold date axis."
            )


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        "**Read the fine print.** The headline 0.86 Sharpe is a point estimate "
        "over 618 OOS days. Lo's HAC t-stat, the block-bootstrap 95% CI, and the "
        "deflated Sharpe all point to the same conclusion: **the result is not "
        "statistically significant at the 5% level.** Treat the absolute number "
        "as suggestive, not as evidence of an edge. Full treatment in "
        "**RESEARCH_NOTE.md §4–§5**."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    start, end, rolling_window, vol_window = render_sidebar()
    render_header()
    render_overview(start, end, rolling_window, vol_window)
    render_rolling(start, end, rolling_window, vol_window)
    render_exposure(start, end)
    render_subperiods()
    render_significance()
    render_drawdown_calendar()
    render_per_ticker()
    render_footer()


if __name__ == "__main__":
    main()
else:  # pragma: no cover -- Streamlit entrypoint
    main()
