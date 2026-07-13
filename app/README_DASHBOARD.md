# Dashboard (Streamlit / Hugging Face Spaces)

Interactive tearsheet over the artifacts emitted by `src/backtest/run.py` and
`src/robustness.py`. **Read-only**: nothing here re-trains models or
re-runs the backtest.

## Cấu trúc

```
app/
├── app.py          # Streamlit entrypoint
├── data_loader.py  # Đọc reports/*.json + reports/*.csv
└── cache.py        # st.cache_data shim

space/
├── README.md       # Cấu hình HF Spaces SDK=streamlit
└── requirements.txt# Pin deps gọn cho Space image
```

> **TF-IDF / LeanStack note:** Dashboard không cần vnstock, torch, xgboost,
> mlflow, quantstats... Tất cả đều được gom lại trong `space/requirements.txt`
> gọn nhẹ.

## Chạy local

```bash
pip install -r space/requirements.txt
streamlit run app/app.py
```

Trang mặc định mở ở `http://localhost:8501`.

## Deploy Hugging Face Spaces

1. Tạo Space mới trên HF, SDK chọn **Streamlit**, license MIT.
2. Push repo hiện tại lên (cần thêm `space/requirements.txt` + `space/README.md`
   ở root của Space). Cách nhanh nhất:

   ```bash
   # Trong root project
   hf init quant-momentum-dashboard --space       # nếu dùng huggingface_hub CLI
   cp -r app reports space .gitattributes .
   # app_file trong README Space trỏ tới "app/app.py" - đã sẵn sàng.
   git add . && git commit -m "Initial dashboard" && git push
   ```

3. Sau khi build xong (~30–60s), Space chạy `streamlit run app/app.py` qua
   cổng mặc định 7860.

### Hai điểm cần lưu ý khi deploy

* **Bố cục thư mục**: Streamlit SDK chạy `app_file` tại root repo. Trong
  `space/README.md` đặt `app_file: app/app.py`. Nếu muốn copy thẳng
  `app.py` lên root thì đổi `app_file: app.py`.
* **Dữ liệu**: Space được cache. Khi `reports/` đổi, **restart Space** (hoặc
  đợi `@st.cache_data` hết hạn) để thấy số mới.

## Tính năng dashboard

| Section | Widget tương tác |
| --- | --- |
| Header | 6 metric tổng (CAGR / Sharpe / Sortino / Max DD vs buy-hold) |
| 1. Equity curve & drawdown | Date **slider** (kéo để zoom sub-period); vẽ strategy vs equal-weight buy-hold |
| 2. Rolling diagnostics | 2 slider (Sharpe window, Vol window, 21–252 ngày) |
| 3. Exposure & turnover | Đếm số ngày off, tổng turnover, time-in-market |
| 4. Sub-period breakdown | 3 bảng: năm, fold, regime |
| 5. Statistical significance | 5 metric: Naive t, Lo HAC, Bootstrap CI, DSR×2 |
| **5b. Drawdown calendar** | **Monthly returns heatmap (year × month) + underwater heatmap + Top-5 drawdown episodes** |
| 6. Per-ticker so sánh | **Multiselect ticker** + 2 bar chart + 1 bảng + line chart overlay strategy vs từng ticker |

Sidebar chứa mọi control. Các tab trên đều là derived figures — refresh chỉ
bằng cách load lại trang.

### Tương tác nâng cao

- **Date range slider** ở sidebar thay vì hai date_input — thân thiện mobile
  và cho phép kéo nhanh tới một sub-period.
- **Multiselect ticker** ở section 6 chọn nhóm mã VN30 để so sánh; chart và
  bảng refresh in-place. Khi `data/processed/*.parquet` tồn tại (chạy
  `python -m src.ingestion`), sẽ có thêm **line chart overlay** giữa strategy
  và buy-and-hold từng mã.
- **Drawdown calendar (section 5b)**: heatmap monthly returns + underwater
  intensity strip + Top-5 drawdown episodes. Dùng Altair (pinned trong
  `space/requirements.txt`) nên không cần thêm deps.
- **Cache** qua `@st.cache_data` (file mtime) — không cần restart app khi
  `reports/` đổi.

## Tests

Helper layer có unit test thuần (không cần Streamlit):

```bash
pip install pytest
python -m pytest tests/ -v
```

Hiện tại 15 test cho `app/data_loader.py` (pure helpers, JSON loaders, và
một sanity check trên `reports/` thật nếu có).

## Refresh data

```bash
# 1. Re-run pipeline upstream
python -m src.backtest.run
python -m src.robustness

# 2. (Space) restart container — hoặc rename `reports/` để vượt cache.

# 3. (Local) F5 hoặc rerun streamlit
```

Mọi logic trong `app/` đã defensive-load: thiếu một file JSON sẽ chỉ ẩn
section đó chứ không crash toàn trang.
