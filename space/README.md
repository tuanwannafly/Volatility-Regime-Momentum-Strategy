---
title: VN30 Regime Momentum Dashboard
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive tearsheet over a volatility-regime momentum backtest on VN30.
---

# VN30 Regime Momentum — Backtest Dashboard

This Space is an **interactive tearsheet** over the artifacts produced by the
[`quant-momentum-regime`](https://github.com/) project:

* walk-forward OOS equity curve
* portfolio & buy-hold summary metrics
* sub-period robustness (year, fold, vol regime)
* statistical significance of the headline Sharpe (Lo t-stat, block-bootstrap,
  deflated Sharpe)

It does **not** re-run any backtest. To refresh the numbers you have to
re-execute the upstream pipeline locally and push the new `reports/` folder.

## Files in this Space

| Path                         | What it is                                 |
| ---------------------------- | ------------------------------------------ |
| `app/app.py`                 | Streamlit entrypoint                       |
| `app/data_loader.py`         | Reads `reports/*.json` & `*.csv`           |
| `app/cache.py`               | `@st.cache_data` shim                      |
| `reports/`                   | JSON/CSV artifacts (data the dashboard reads) |
| `space/requirements.txt`     | Pinned runtime deps (Streamlit SDK)        |
| `space/README.md`            | This file                                  |

The dashboard `app_file` is `app/app.py` (Streamlit runs it from the repo
root). If you'd rather serve the dashboard as a standalone Space, copy
`app/app.py` → `app.py` at the repository root and update `app_file`
accordingly.

## Local

```bash
pip install -r space/requirements.txt
streamlit run app/app.py
```

## Refresh

Replace `reports/` with a freshly generated one (run `python -m src.backtest.run`
and `python -m src.robustness` upstream) and restart the Space. The cached
values are tied to file mtime by `@st.cache_data`.
