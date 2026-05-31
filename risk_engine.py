"""
risk_engine.py
==============
Portfolio VaR / CVaR / stress-testing engine for the IEOR 4703 term project.

This module is a faithful extraction of the analysis logic from
``IEOR_4703_Term_Project.ipynb`` so the Streamlit UI (``app.py``) produces
results identical to the notebook. Nothing here imports Streamlit or
matplotlib — it is pure data + math, so it can be reused/tested anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants (match the notebook) ────────────────────────────────────────────
DATA_DIR      = Path(__file__).resolve().parent / "data"
STUDY_START   = pd.Timestamp("2005-12-31")
STUDY_END     = pd.Timestamp("2024-12-31")
MAX_FFILL     = 5
INITIAL_VALUE = 1_000_000          # capital on behalf of the client ($)
DEFAULT_DUR, DEFAULT_YTM = 4.65, 0.0287   # global-median fallbacks for bonds

# Populated by load_market_data(); referenced by the bond imputer.
BOND_DUR: pd.Series = pd.Series(dtype=float)
BOND_YTM: pd.Series = pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  Monte-Carlo price imputation  (verbatim from the notebook)
# ══════════════════════════════════════════════════════════════════════════════
def _gbm_params(series):
    lr = np.log(series / series.shift(1)).dropna()
    if len(lr) < 20:
        return 0.0, 0.02
    return float(lr.mean()), float(lr.std())


def _gap_bounds(vals):
    """Return (starts, ends) index arrays of each contiguous NaN run."""
    is_nan  = np.isnan(vals)
    changes = np.diff(is_nan.astype(int), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0] - 1
    return starts, ends


def _brownian_bridge(vals, s, e, sigma, rng):
    """Fill interior gap [s, e] by a Brownian Bridge in log-price."""
    log_a, log_b = np.log(vals[s - 1]), np.log(vals[e + 1])
    m = e - s + 2
    for k, idx in enumerate(range(s, e + 1)):
        t = k + 1
        bb_mean = log_a + (log_b - log_a) * t / m
        bb_std  = sigma * np.sqrt(t * (m - t) / m)
        vals[idx] = np.exp(bb_mean + bb_std * rng.standard_normal())


def impute_prices_mc(prices_series, max_ffill=MAX_FFILL, rng=None):
    """Impute missing EQUITY prices: ffill / Brownian Bridge / forward GBM."""
    if rng is None:
        rng = np.random.default_rng(42)
    p = prices_series.ffill(limit=max_ffill)
    if not p.isna().any():
        return p
    mu, sigma = _gbm_params(p.dropna())
    vals = p.to_numpy(dtype=float).copy()
    n    = len(vals)
    starts, ends = _gap_bounds(vals)
    for s, e in zip(starts, ends):
        has_L = s > 0     and not np.isnan(vals[s - 1])
        has_R = e < n - 1 and not np.isnan(vals[e + 1])
        if has_L and has_R:                        # Brownian Bridge
            _brownian_bridge(vals, s, e, sigma, rng)
        elif has_L:                                # Forward GBM
            S = vals[s - 1]
            for idx in range(s, e + 1):
                S = S * np.exp((mu - 0.5 * sigma**2) + sigma * rng.standard_normal())
                vals[idx] = S
    return pd.Series(vals, index=prices_series.index, name=prices_series.name)


def impute_bond_prices_mc(prices_series, fid, max_ffill=MAX_FFILL, rng=None, kappa=0.02):
    """Impute missing BOND prices: trailing gaps use a duration-carry +
    mean-reverting (Vasicek) yield model instead of GBM."""
    if rng is None:
        rng = np.random.default_rng(42)
    p = prices_series.ffill(limit=max_ffill)
    if not p.isna().any():
        return p
    _, sigma = _gbm_params(p.dropna())                 # bond's own daily price vol
    D     = BOND_DUR.get(fid, DEFAULT_DUR) or DEFAULT_DUR
    theta = BOND_YTM.get(fid, DEFAULT_YTM) or DEFAULT_YTM
    sig_y = sigma / max(D, 0.5)                         # implied yield vol
    vals = p.to_numpy(dtype=float).copy()
    n    = len(vals)
    starts, ends = _gap_bounds(vals)
    for s, e in zip(starts, ends):
        has_L = s > 0     and not np.isnan(vals[s - 1])
        has_R = e < n - 1 and not np.isnan(vals[e + 1])
        if has_L and has_R:                            # Brownian Bridge
            _brownian_bridge(vals, s, e, sigma, rng)
        elif has_L:                                    # Duration-carry + Vasicek yield
            S, y = vals[s - 1], theta
            for idx in range(s, e + 1):
                dy  = kappa * (theta - y) + sig_y * rng.standard_normal()
                ret = y / 252.0 - D * dy               # carry minus rate move
                y  += dy
                S  *= (1.0 + ret)
                vals[idx] = S
    return pd.Series(vals, index=prices_series.index, name=prices_series.name)


# ══════════════════════════════════════════════════════════════════════════════
#  Market-data loading  (one-shot; cache this in the app)
# ══════════════════════════════════════════════════════════════════════════════
def load_market_data():
    """Load prices, impute gaps, build the top-50 fund pool, and load the
    holdings files used for stress testing. Returns a dict consumed by the UI."""
    global BOND_DUR, BOND_YTM

    eq_prices   = pd.read_csv(DATA_DIR / "us_equity_adj_close.csv",
                              index_col="as_of", parse_dates=True)
    bond_prices = pd.read_csv(DATA_DIR / "us_bond_intermediate_core_adj_close.csv",
                              index_col="as_of", parse_dates=True)

    all_prices = pd.concat([eq_prices, bond_prices], axis=1).sort_index()
    all_prices = all_prices.loc[STUDY_START:STUDY_END]

    # Candidate pool: funds with price data from the study start
    early_window   = all_prices.loc["2005-12-31":"2006-01-15"]
    candidate_pool = early_window.columns[early_window.notna().any()].tolist()

    # Rank by most-recent AUM, keep top 50
    eq_size   = pd.read_csv(DATA_DIR / "us_equity_fund_size.csv")
    bond_size = pd.read_csv(DATA_DIR / "us_bond_intermediate_fund_size.csv")
    all_sizes = pd.concat([eq_size, bond_size], ignore_index=True)
    latest_aum = (all_sizes.sort_values("as_of")
                            .groupby("ask_id")["fund_size"]
                            .last()
                            .reindex(candidate_pool)
                            .fillna(0.0))
    fund_pool = latest_aum.sort_values(ascending=False).head(50).index.tolist()

    # Per-bond duration & yield (for the bond imputer)
    _dy = pd.read_csv(DATA_DIR / "us_bond_intermediate_fixed_income_duration_yield.csv")
    BOND_DUR = _dy.groupby("ask_id")["modified_duration"].median()
    BOND_YTM = _dy.groupby("ask_id")["yield_to_maturity"].median() / 100.0

    # Impute (equity → GBM/bridge, bond → duration-carry) — same seed as notebook
    bond_cols  = set(bond_prices.columns)
    rng_impute = np.random.default_rng(0)
    all_prices = all_prices.apply(
        lambda col: impute_bond_prices_mc(col, col.name, rng=rng_impute)
                    if col.name in bond_cols
                    else impute_prices_mc(col, rng=rng_impute)
    )

    # Metadata
    eq_meta   = pd.read_csv(DATA_DIR / "us_equity_intermediate_meta_data.csv",
                            index_col="ask_id")[["sec_name", "ticker"]]
    bond_meta = pd.read_csv(DATA_DIR / "us_bond_intermediate_meta_data.csv",
                            index_col="ask_id")[["sec_name", "ticker"]]
    metadata  = pd.concat([eq_meta, bond_meta])

    # Holdings files for stress testing
    eq_sectors   = pd.read_csv(DATA_DIR / "us_equity_sectors.csv")
    bond_sectors = pd.read_csv(DATA_DIR / "us_bond_intermediate_fixed_income_primary_sector.csv")
    bond_dur     = pd.read_csv(DATA_DIR / "us_bond_intermediate_fixed_income_duration_yield.csv")
    bond_credit  = pd.read_csv(DATA_DIR / "us_bond_intermediate_core_credit_quality.csv")
    eq_region    = pd.read_csv(DATA_DIR / "us_equity_economic_region_exposure.csv")

    return {
        "all_prices":   all_prices,
        "fund_pool":    fund_pool,
        "metadata":     metadata,
        "latest_aum":   latest_aum,
        "eq_cols":      set(eq_prices.columns),
        "bond_cols":    bond_cols,
        # stress holdings
        "eq_sectors":   eq_sectors,
        "bond_sectors": bond_sectors,
        "bond_dur":     bond_dur,
        "bond_credit":  bond_credit,
        "eq_region":    eq_region,
    }


def _pretty_sector(col):
    """'equity_econ_sector_consumer_cyclical_pct_net' -> 'Consumer Cyclical'."""
    name = col.replace("equity_econ_sector_", "").replace("_pct_net", "")
    return name.replace("_", " ").title()


def _pretty_rating(col):
    """'credit_qual_below_b_pct' -> 'Below B';  'credit_qual_aaa_pct' -> 'AAA'."""
    name = col.replace("credit_qual_", "").replace("_pct", "")
    special = {"below_b": "Below B", "not_rated": "Not Rated"}
    if name in special:
        return special[name]
    return name.upper()


def fund_classifications(data):
    """Classify each fund for the sidebar filters:
      * equity funds  -> primary (largest-weight) economic sector
      * bond funds    -> dominant (largest-weight) credit-quality bucket
    Returns dict with per-fund maps and the sorted lists of distinct labels."""
    eq, bc = data["eq_sectors"], data["bond_credit"]
    eq_cols = [c for c in eq.columns if c.startswith("equity_econ_sector")]
    cr_cols = [c for c in bc.columns if c.startswith("credit_qual")]

    eq_latest = eq.sort_values("as_of").groupby("ask_id")[eq_cols].last()
    primary_sector = {}
    for fid, row in eq_latest.iterrows():
        vals = row.astype(float).fillna(0.0)
        if vals.sum() > 0:
            primary_sector[fid] = _pretty_sector(vals.idxmax())

    cr_latest = bc.sort_values("as_of").groupby("ask_id")[cr_cols].last()
    dom_rating = {}
    for fid, row in cr_latest.iterrows():
        vals = row.astype(float).fillna(0.0)
        if vals.sum() > 0:
            dom_rating[fid] = _pretty_rating(vals.idxmax())

    return {
        "primary_sector": primary_sector,
        "dom_rating":     dom_rating,
        "sectors":        sorted(set(primary_sector.values())),
        "ratings":        sorted(set(dom_rating.values())),
    }


def fund_label(fid, metadata):
    """Human-readable label for a fund id, e.g. 'FLPSX — Fidelity Low-Priced Stock'."""
    if fid in metadata.index:
        tick = metadata.loc[fid, "ticker"]
        name = metadata.loc[fid, "sec_name"]
        tick = "" if pd.isna(tick) else str(tick)
        return f"{tick}  —  {name}".strip(" —")
    return fid


# ══════════════════════════════════════════════════════════════════════════════
#  Core analysis  (verbatim from the notebook)
# ══════════════════════════════════════════════════════════════════════════════
def build_rolling_returns(prices_df, fund_ids, ts, te, tau, delta):
    """Rolling τ-horizon returns for an equal-weight portfolio."""
    p = prices_df.loc[pd.Timestamp(ts):pd.Timestamp(te), fund_ids].copy()
    p = p.ffill()

    N          = len(p)
    prices_arr = p.to_numpy(dtype=float)

    start_idx = np.arange(0, N - tau, delta)
    end_idx   = start_idx + tau
    L         = len(start_idx)

    p0 = prices_arr[start_idx]
    pt = prices_arr[end_idx]
    fund_returns      = (pt - p0) / p0
    portfolio_returns = np.nanmean(fund_returns, axis=1)

    return fund_returns, portfolio_returns, p.index[start_idx], L


def compute_var_cvar(portfolio_returns, alpha, n_sim, initial_value, seed=0):
    """Bootstrap simulation of VaR and CVaR (dollar losses, positive = bad)."""
    rng = np.random.default_rng(seed)
    sim = rng.choice(portfolio_returns, size=n_sim, replace=True)

    q           = np.percentile(sim, (1 - alpha) * 100)
    var_dollar  = -q * initial_value
    tail_mean   = sim[sim <= q].mean()
    cvar_dollar = -tail_mean * initial_value

    return {
        "VaR":         var_dollar,
        "CVaR":        cvar_dollar,
        "VaR_return":  q,
        "CVaR_return": tail_mean,
        "sim":         sim,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Stress testing  (verbatim from the notebook)
# ══════════════════════════════════════════════════════════════════════════════
def _latest_row(df, fund_ids, value_cols, id_col="ask_id", date_col="as_of"):
    rows = []
    for fid in fund_ids:
        sub = df[df[id_col] == fid].sort_values(date_col)
        row = (sub.iloc[-1][value_cols].astype(float).fillna(0.0)
               if not sub.empty
               else pd.Series(0.0, index=value_cols))
        rows.append(row.rename(fid))
    return pd.DataFrame(rows).astype(float)


def prepare_stress_inputs(data, selected_funds):
    """Build the per-fund exposure tables needed by compute_fund_shocks."""
    eq_sectors, bond_sectors = data["eq_sectors"], data["bond_sectors"]
    bond_dur, bond_credit, eq_region = data["bond_dur"], data["bond_credit"], data["eq_region"]

    eq_sec_cols   = [c for c in eq_sectors.columns   if c.startswith("equity_econ_sector")]
    bond_sec_cols = [c for c in bond_sectors.columns if c.startswith("fixed_inc_ps")]
    credit_cols   = [c for c in bond_credit.columns  if c.startswith("credit_qual")]
    region_cols   = [c for c in eq_region.columns    if c.startswith("equity_region")]

    eq_funds   = [f for f in selected_funds if f in eq_sectors["ask_id"].values]
    bond_funds = [f for f in selected_funds if f in bond_sectors["ask_id"].values]

    eq_w     = _latest_row(eq_sectors,  eq_funds,   eq_sec_cols)
    bond_w   = _latest_row(bond_sectors, bond_funds, bond_sec_cols)
    credit_w = _latest_row(bond_credit,  bond_funds, credit_cols)
    region_w = _latest_row(eq_region,    eq_funds,   region_cols)

    dur_df  = bond_dur.sort_values("as_of").dropna(subset=["modified_duration"])
    mod_dur = (dur_df.groupby("ask_id")["modified_duration"]
                     .last().reindex(bond_funds).fillna(0.0))

    return dict(eq_w=eq_w, bond_w=bond_w, credit_w=credit_w,
                region_w=region_w, mod_dur=mod_dur,
                eq_funds=eq_funds, bond_funds=bond_funds)


def compute_fund_shocks(fund_ids, eq_w, bond_w, credit_w, region_w, mod_dur,
                        eq_sec_shocks=None, bond_sec_shocks=None,
                        rate_shock_bps=0.0, credit_shocks=None, region_shocks=None):
    """Per-fund additive return shock from five exposure channels."""
    eq_sec_shocks   = eq_sec_shocks   or {}
    bond_sec_shocks = bond_sec_shocks or {}
    credit_shocks   = credit_shocks   or {}
    region_shocks   = region_shocks   or {}
    delta_r = rate_shock_bps / 10_000.0

    shocks = {}
    for fid in fund_ids:
        c = 0.0
        if fid in eq_w.index:                              # 1. equity sector
            for col, s in eq_sec_shocks.items():
                c += (eq_w.loc[fid, col] / 100.0) * s
        if fid in bond_w.index:                            # 2. bond sector
            for col, s in bond_sec_shocks.items():
                c += (bond_w.loc[fid, col] / 100.0) * s
        if fid in mod_dur.index:                           # 3. rate: ΔP/P ≈ -D·Δr
            c += -mod_dur[fid] * delta_r
        if fid in credit_w.index:                          # 4. credit spread
            for col, s in credit_shocks.items():
                c += (credit_w.loc[fid, col] / 100.0) * s
        if fid in region_w.index:                          # 5. geographic
            for col, s in region_shocks.items():
                c += (region_w.loc[fid, col] / 100.0) * s
        shocks[fid] = c
    return shocks


def apply_shocks(fund_returns_arr, fund_ids, shocks):
    """Add per-fund shock scalars to returns → equal-weight portfolio series."""
    stressed = fund_returns_arr.copy()
    for j, fid in enumerate(fund_ids):
        stressed[:, j] += shocks.get(fid, 0.0)
    return np.nanmean(stressed, axis=1)


# Stress scenarios: (label, kwargs for compute_fund_shocks)  — same as notebook
SCENARIOS = [
    ("Tech Crash −30%",
     dict(eq_sec_shocks={"equity_econ_sector_technology_pct_net": -0.30})),
    ("Energy Shock −40%",
     dict(eq_sec_shocks={"equity_econ_sector_energy_pct_net": -0.40})),
    ("Rate Spike +200 bps",
     dict(rate_shock_bps=200)),
    ("Credit Crisis\nIG +300bps / HY +600bps",
     dict(credit_shocks={
         "credit_qual_aaa_pct":     -0.01,
         "credit_qual_aa_pct":      -0.02,
         "credit_qual_a_pct":       -0.03,
         "credit_qual_bbb_pct":     -0.05,
         "credit_qual_bb_pct":      -0.08,
         "credit_qual_b_pct":       -0.12,
         "credit_qual_below_b_pct": -0.18,
     })),
    ("EM Crisis\nEmerging Markets −25%",
     dict(region_shocks={
         "equity_region_emerging_pct_net":      -0.25,
         "equity_region_asia_emrg_pct_net":     -0.25,
         "equity_region_latin_america_pct_net": -0.25,
         "equity_region_europe_emrg_pct_net":   -0.20,
     })),
    ("Broad Crisis\n(all channels)",
     dict(eq_sec_shocks={
              "equity_econ_sector_technology_pct_net":         -0.25,
              "equity_econ_sector_financial_services_pct_net": -0.20,
              "equity_econ_sector_consumer_cyclical_pct_net":  -0.18,
          },
          bond_sec_shocks={"fixed_inc_ps_corporate_bond_pct_net": -0.10},
          rate_shock_bps=150,
          credit_shocks={
              "credit_qual_bbb_pct": -0.04,
              "credit_qual_bb_pct":  -0.07,
              "credit_qual_b_pct":   -0.10,
          },
          region_shocks={"equity_region_emerging_pct_net": -0.20})),
]


# ── Parametric scenario builder ───────────────────────────────────────────────
# Lets the UI set different values per scenario. The defaults below exactly
# reproduce the static SCENARIOS list above (tech −30, energy −40, rate +200,
# credit IG/HY ×1, EM −25, broad ×1 + rate +150).
DEFAULT_SCENARIO_CONFIG = {
    "tech":   {"enabled": True, "tech_pct":   -30.0},
    "energy": {"enabled": True, "energy_pct": -40.0},
    "rate":   {"enabled": True, "rate_bps":   200.0},
    "credit": {"enabled": True, "ig_mult": 1.0, "hy_mult": 1.0},
    "em":     {"enabled": True, "em_pct":     -25.0},
    "broad":  {"enabled": True, "intensity": 1.0, "rate_bps": 150.0},
}


def build_scenarios(cfg=None):
    """Build a [(label, kwargs)] scenario list from a UI config dict.
    Disabled scenarios are skipped; magnitudes come from the config."""
    cfg = cfg or DEFAULT_SCENARIO_CONFIG
    out = []

    t = cfg["tech"]
    if t["enabled"]:
        out.append((f"Tech Crash {t['tech_pct']:+.0f}%",
                    dict(eq_sec_shocks={"equity_econ_sector_technology_pct_net": t["tech_pct"] / 100})))

    e = cfg["energy"]
    if e["enabled"]:
        out.append((f"Energy Shock {e['energy_pct']:+.0f}%",
                    dict(eq_sec_shocks={"equity_econ_sector_energy_pct_net": e["energy_pct"] / 100})))

    r = cfg["rate"]
    if r["enabled"]:
        out.append((f"Rate Spike {r['rate_bps']:+.0f} bps",
                    dict(rate_shock_bps=r["rate_bps"])))

    c = cfg["credit"]
    if c["enabled"]:
        ig, hy = c["ig_mult"], c["hy_mult"]
        out.append((f"Credit Crisis\nIG×{ig:.1f} / HY×{hy:.1f}",
                    dict(credit_shocks={
                        "credit_qual_aaa_pct":     -0.01 * ig,
                        "credit_qual_aa_pct":      -0.02 * ig,
                        "credit_qual_a_pct":       -0.03 * ig,
                        "credit_qual_bbb_pct":     -0.05 * ig,
                        "credit_qual_bb_pct":      -0.08 * hy,
                        "credit_qual_b_pct":       -0.12 * hy,
                        "credit_qual_below_b_pct": -0.18 * hy,
                    })))

    m = cfg["em"]
    if m["enabled"]:
        v = m["em_pct"] / 100
        out.append((f"EM Crisis {m['em_pct']:+.0f}%",
                    dict(region_shocks={
                        "equity_region_emerging_pct_net":      v,
                        "equity_region_asia_emrg_pct_net":     v,
                        "equity_region_latin_america_pct_net": v,
                        "equity_region_europe_emrg_pct_net":   v * 0.8,
                    })))

    b = cfg["broad"]
    if b["enabled"]:
        k = b["intensity"]
        out.append((f"Broad Crisis ×{k:.1f}",
                    dict(eq_sec_shocks={
                             "equity_econ_sector_technology_pct_net":         -0.25 * k,
                             "equity_econ_sector_financial_services_pct_net": -0.20 * k,
                             "equity_econ_sector_consumer_cyclical_pct_net":  -0.18 * k,
                         },
                         bond_sec_shocks={"fixed_inc_ps_corporate_bond_pct_net": -0.10 * k},
                         rate_shock_bps=b["rate_bps"],
                         credit_shocks={
                             "credit_qual_bbb_pct": -0.04 * k,
                             "credit_qual_bb_pct":  -0.07 * k,
                             "credit_qual_b_pct":   -0.10 * k,
                         },
                         region_shocks={"equity_region_emerging_pct_net": -0.20 * k})))

    return out


def run_stress(fund_returns, selected_funds, stress_inputs, base_var, base_cvar,
               alpha, n_sim, initial_value, scenarios=None):
    """Run all scenarios; return a list of dicts with stressed VaR/CVaR and deltas."""
    scenarios = scenarios if scenarios is not None else SCENARIOS
    out = []
    for label, kwargs in scenarios:
        shocks      = compute_fund_shocks(selected_funds,
                                          stress_inputs["eq_w"], stress_inputs["bond_w"],
                                          stress_inputs["credit_w"], stress_inputs["region_w"],
                                          stress_inputs["mod_dur"], **kwargs)
        stressed_pr = apply_shocks(fund_returns, selected_funds, shocks)
        s           = compute_var_cvar(stressed_pr, alpha, n_sim, initial_value)
        out.append({
            "label": label,
            "VaR":   s["VaR"],   "CVaR":  s["CVaR"],
            "dVaR":  s["VaR"] - base_var,
            "dCVaR": s["CVaR"] - base_cvar,
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience wrappers used by the UI
# ══════════════════════════════════════════════════════════════════════════════
DELTA_LABEL = {1: "daily", 5: "weekly", 22: "monthly", 66: "quarterly"}


def run_analysis(data, selected_funds, ts, te, tau, delta, alpha,
                 n_sim, initial_value=INITIAL_VALUE):
    """Full single-portfolio analysis. Returns everything the UI needs to render."""
    fund_returns, portfolio_returns, start_dates, L = build_rolling_returns(
        data["all_prices"], selected_funds, ts, te, tau, delta)
    results = compute_var_cvar(portfolio_returns, alpha, n_sim, initial_value)
    return {
        "fund_returns":      fund_returns,
        "portfolio_returns": portfolio_returns,
        "start_dates":       start_dates,
        "L":                 L,
        "results":           results,
    }


def sweep_d(data, fund_pool, ts, te, tau, delta, alpha, n_sim, initial_value):
    """Diversification curve: VaR/CVaR for random equal-weight portfolios of
    size d drawn from the 50-fund pool (matches notebook Step 4a)."""
    rng_sweep = np.random.default_rng(0)
    max_d     = min(len(fund_pool), 50)
    d_values  = sorted(set([1, 2, 3, 5] + list(range(5, max_d + 1, max(1, max_d // 12)))))
    vars_d, cvars_d = [], []
    for d_val in d_values:
        funds_sel = list(rng_sweep.choice(fund_pool, size=d_val, replace=False))
        _, pr, _, _ = build_rolling_returns(data["all_prices"], funds_sel, ts, te, tau, delta)
        res = compute_var_cvar(pr, alpha, n_sim, initial_value, seed=d_val)
        vars_d.append(res["VaR"]); cvars_d.append(res["CVaR"])
    return d_values, vars_d, cvars_d


def sweep_delta(data, selected_funds, ts, te, tau, alpha, n_sim, initial_value):
    delta_sweep = [1, 5, 22, 66]
    vars_, cvars_, Ls = [], [], []
    for dlt in delta_sweep:
        _, pr, _, L_v = build_rolling_returns(data["all_prices"], selected_funds, ts, te, tau, dlt)
        res = compute_var_cvar(pr, alpha, n_sim, initial_value, seed=dlt)
        vars_.append(res["VaR"]); cvars_.append(res["CVaR"]); Ls.append(L_v)
    return delta_sweep, vars_, cvars_, Ls


def sweep_tau(data, selected_funds, ts, te, delta, alpha, n_sim, initial_value):
    tau_sweep  = [63, 126, 252, 504, 756, 1260]
    tau_labels = ["3 mo", "6 mo", "1 yr", "2 yr", "3 yr", "5 yr"]
    vars_, cvars_ = [], []
    for t_val in tau_sweep:
        _, pr, _, L_v = build_rolling_returns(data["all_prices"], selected_funds, ts, te, t_val, delta)
        if L_v < 5:
            vars_.append(np.nan); cvars_.append(np.nan); continue
        res = compute_var_cvar(pr, alpha, n_sim, initial_value, seed=t_val)
        vars_.append(res["VaR"]); cvars_.append(res["CVaR"])
    return tau_sweep, tau_labels, vars_, cvars_