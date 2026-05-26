# WTI Crude Oil Price Forecaster

**ARIMA vs Prophet: which model predicts oil prices better?**

**[Live Dashboard →](https://izu6qfxgzu4cgjhnhcuxh7.streamlit.app/)**

25 years of WTI Crude Oil futures (CL=F) — 2000 to 2024. Trains ARIMA(1,1,1) and Facebook Prophet on 2000–2022 data, then evaluates both on a 24-month out-of-sample forecast (2023–2024).

---

## Results

| Model | MAE ($/barrel) | RMSE ($/barrel) | Verdict |
|---|---|---|---|
| **ARIMA(1,1,1)** | **$5.29** | **$6.77** | Winner |
| Prophet | $5.72 | $6.85 | — |

ARIMA wins on both error metrics. Both models missed the 2022–2023 price decline from ~$90 to ~$70 — a geopolitical de-escalation not visible in historical price patterns.

---

## Key Findings

- **ARIMA outperforms Prophet by $0.43/barrel MAE** — mean-reversion performs well for oil because OPEC production targets and breakeven costs keep prices from drifting indefinitely
- **Both models struggled with geopolitical shocks** — the 2022–2023 decline driven by de-escalation expectations and demand slowdown is invisible to statistical models
- **Prophet adds interpretable components ARIMA lacks** — trend decomposition and yearly seasonality patterns are economically meaningful even with slightly lower accuracy
- **Oil prices show a mild summer demand premium** — Prophet's seasonal component confirms the driving season pattern (June–August stronger) and Q4 softening
- **$5–7/barrel MAE over 24 months is reasonable** — WTI annualized volatility is ~35–40%; both models outperform a naive last-price baseline

---

## Dashboard Tabs

| Tab | What It Shows |
|---|---|
| Price History | 25-year daily price with annotated key events + 365-day moving average |
| ARIMA Forecast | ARIMA(1,1,1) forecast vs actual with 95% confidence interval |
| Prophet Forecast | Prophet forecast vs actual + trend and seasonality components |
| Model Comparison | Side-by-side overlay + error metric bar chart + comparison table |

---

## Data

**Source:** Yahoo Finance via `yfinance` — WTI Crude Oil Futures (CL=F)

| Series | Frequency | Period |
|---|---|---|
| Daily closing price | Daily | 2000–2024 |
| Monthly resampled | Monthly (MS) | 2000–2024 |

---

## Run Locally

```bash
git clone https://github.com/favigarcia1629/oil.git
cd oil
pip install -r requirements.txt
streamlit run app.py
```

No API keys needed — yfinance pulls all data automatically.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| yfinance | WTI Crude Oil futures data |
| statsmodels | ARIMA model fitting and forecasting |
| Prophet (Facebook) | Trend + seasonality decomposition forecasting |
| scikit-learn | MAE and RMSE error metrics |
| Plotly | Interactive charts |
| Streamlit | 4-tab dashboard |

*Data: Yahoo Finance. Not financial advice — built for research and education.*
