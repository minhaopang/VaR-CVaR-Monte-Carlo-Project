# Mutual-Fund Portfolio Risk Tool — VaR, CVaR & Stress Testing

**IEOR 4703 — Monte Carlo Simulation Methods (Hirsa) · Term Project**

Minhao (Myron) Pang · Siyun (Cheryl) Yan

A tool that lets a portfolio manager invest **$1,000,000 on behalf of a client**,
spread **equally** across any **d** mutual funds (1 ≤ d ≤ 50) chosen from a pool of 50,
and measure the risk of that portfolio over a chosen investment horizon.

It answers three questions:

1. **VaR (Value-at-Risk)** — "In a bad year, how much could we lose?"
2. **CVaR (Conditional VaR)** — "If that bad case happens, how bad is the *average* loss?"
3. **Stress testing** — "What happens to those losses under a tech crash, a rate spike, a credit crisis, etc.?"

There are **two ways** to use the project:

- **A point-and-click web app** (`app.py`) — recommended for a portfolio manager.
- **The original Jupyter notebook** (`IEOR_4703_Term_Project.ipynb`) — the full write-up with all the math, for review.

---

## Quick start — launch the web app

You need Python 3.9+.

```bash
# 1. from the project folder, install the dependencies (one time)
python3 -m pip install -r requirements.txt

# 2. launch the app
python3 -m streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`. 

> The first load takes a few seconds because the app reads the price history and fills in missing
> prices. After that it's instant; results update as soon as you change any control.

---

## What's in each file

| File | What it does |
|------|--------------|
| **`app.py`** | The web dashboard the portfolio manager uses. All the buttons, dropdowns, charts, and the report download live here. It does no math itself — it calls `risk_engine.py`. |
| **`risk_engine.py`** | The "calculation engine." Loads the data, fills missing prices, builds the return distributions, and computes VaR, CVaR, and the stress scenarios. This is the same logic as the notebook, packaged so the app can reuse it. |
| **`IEOR_4703_Term_Project.ipynb`** | The original Jupyter notebook — the full step-by-step analysis and explanations for the d = 10 example portfolio. Read this to understand the methodology. |
| **`requirements.txt`** | The list of Python packages the project needs. |
| **`data/`** | The provided dataset: daily prices for the 50 funds plus their holdings (sectors, credit quality, duration/yield, region, fund size). The app reads everything from here. |

---

## Using the web app

Everything is driven from the **left sidebar**; results appear on the right and update live.

### 1. Pick the portfolio

- **Search box (🔍)** — type any part of a fund's name or ticker (e.g. `vangu`, `FLPSX`) and the
  matching fund appears immediately.
- **Asset type** — show All funds, only Equity, or only Bond.
- **Equity primary sector** — narrow equity funds to a sector (Technology, Healthcare, …), based on
  each fund's largest sector holding.
- **Bond dominant credit rating** — narrow bond funds by their main credit-quality bucket (AAA, BBB, …).
- **Selected funds** — the multi-select list. Pick **any number** of funds; **d** is simply how many
  you picked. **➕ Add filtered** adds everything matching the current filter; **✖ Clear all** empties it.

Capital is always **$1,000,000 split equally** across the d funds you choose — so each fund gets `1/d`
of the money, exactly as the assignment specifies.

### 2. Set the risk parameters

| Control | Meaning | Choices |
|---------|---------|---------|
| **Horizon τ** | How far ahead you measure returns | 3 / 6 months, **1** / 2 / 3 / 5 years |
| **Rolling step δ** | How often a historical window starts | Daily, Weekly, **Monthly**, Quarterly |
| **Confidence α** | How extreme a loss VaR/CVaR describe | 90%, **95%**, 99% |
| **Bootstrap draws** | How many simulations to run | 1k – 50k (default 10k) |

### 3. Read the results

- **Results summary** — the headline **VaR** and **CVaR** in dollars (and as a % loss), the average
  return, and the number of historical windows used. An expander shows the exact holdings and
  distribution statistics. *(This mirrors the notebook's Summary section.)*
- **Simulated return distribution** — a histogram color-coded by loss severity, the empirical CDF
  with the VaR/CVaR thresholds marked, and the rolling-return time series.
- **Sensitivity analysis** (toggle) — how VaR/CVaR change as you vary the **number of funds d**
  (the diversification benefit), the **horizon τ**, and the **rolling step δ**.

### 4. Stress test the portfolio

Tick **Run stress scenarios** to see how each crisis scenario raises VaR and CVaR. A scenario applies
a per-fund return shock based on what each fund actually holds (its sectors, credit quality, interest-rate
sensitivity, and region), then re-measures the risk.

Open **⚙️ Customize scenario severities** to set your **own values per scenario** — or untick any to skip it:

| Scenario | What you control |
|----------|------------------|
| **Tech Crash** | size of the technology shock (%) |
| **Energy Shock** | size of the energy shock (%) |
| **Rate Spike** | interest-rate move (basis points) — hits bonds via their duration |
| **EM Crisis** | emerging-markets equity shock (%) |
| **Credit Crisis** | separate severity multipliers for investment-grade (IG) and high-yield (HY) |
| **Broad Crisis** | an overall intensity multiplier across all channels, plus its own rate move |

The defaults reproduce the project's original calibration, so leaving them untouched matches the notebook.

### 5. Export a client report

At the top of the page, **📄 Export client report** gives two downloads:

- **CSV** — the parameters, VaR/CVaR, holdings, and stress-test table (open in Excel).
- **PDF** — a cover page (summary + holdings) followed by every chart, ready to share with the client.

---

## How the numbers are calculated 

1. **Filling gaps in the price history.** Some funds are missing prices on some days. Short gaps are
   carried forward; gaps with a known price on both sides are bridged; open-ended gaps are simulated.
   **Equity** funds are simulated as a random walk (Geometric Brownian Motion); **bond** funds use a
   fixed-income model that accounts for a bond earning its yield and reacting to interest-rate moves
   (so bond prices don't drift unrealistically).
2. **Long-horizon returns.** For each fund we compute the return over the horizon τ, sampled on a
   rolling schedule with step δ — giving L historical observations per fund.
3. **Portfolio returns.** The equally-weighted average of the funds' returns at each window.
4. **VaR & CVaR.** We resample (bootstrap) the historical portfolio returns many times to build the
   loss distribution, then read off VaR (the α-quantile loss) and CVaR (the average loss beyond VaR),
   scaled to the $1,000,000 invested.
5. **Stress testing.** Each scenario shifts the historical returns down by a fund-specific shock and
   the VaR/CVaR are recomputed — showing how much the chosen crisis would add to the risk.


---

## Data

All data lives in `data/` and covers **2005-12-31 → 2024-12-31** for **50 mutual funds**
(equity and intermediate-term US bond funds): daily adjusted-close prices, fund size (AUM), sector and
region exposures, bond credit quality, and bond duration/yield. The tool uses only this provided dataset.

## Requirements

`streamlit`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy` — all pinned in `requirements.txt`.

---

### Troubleshooting

- **`streamlit: command not found`** — use `python3 -m streamlit run app.py` (as shown above); it always works.
- **"too few rolling windows" error** — your horizon τ or step δ is too large for the date range; pick a smaller value.
- **Nothing matches a filter** — clear the search box and set the filters back to *Any* / *All*.