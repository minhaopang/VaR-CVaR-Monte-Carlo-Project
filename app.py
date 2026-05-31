"""
app.py — Portfolio VaR / CVaR / Stress-Testing dashboard
========================================================
A Streamlit front-end over ``risk_engine.py`` for the portfolio manager.

Run with:
    streamlit run app.py

The PM selects any number d of funds from the 50-fund pool; capital is fixed
at $1,000,000 distributed equally. The app shows the results summary plus all
the plots from the notebook (return distribution, sensitivity sweeps, stress).
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import risk_engine as eng

# ── Page + style ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio VaR / CVaR Dashboard",
                   page_icon="📉", layout="wide")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

fmt_dollar = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
fmt_pct    = mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")

TS, TE = "2006-01-03", "2024-12-31"   # study window (fixed)


# ── Cached data load (runs once; imputation is expensive) ──────────────────────
@st.cache_data(show_spinner="Loading prices and imputing missing data …")
def get_data():
    return eng.load_market_data()


data       = get_data()
fund_pool  = data["fund_pool"]
metadata   = data["metadata"]
latest_aum = data["latest_aum"]


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — controls
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️  Portfolio setup")
st.sidebar.caption("Capital fixed at **$1,000,000**, distributed equally.")

# Funds ordered by AUM (largest first) so the menu is sensible
pool_by_aum = latest_aum.reindex(fund_pool).sort_values(ascending=False).index.tolist()
labels = {fid: eng.fund_label(fid, metadata) for fid in pool_by_aum}

default_pick = pool_by_aum[:10]   # the d=10 portfolio the project was tuned on
selected_funds = st.sidebar.multiselect(
    f"Select funds  (pool of {len(fund_pool)})",
    options=pool_by_aum,
    default=default_pick,
    format_func=lambda fid: labels[fid],
    help="Pick any number of funds. d = how many you select.",
)
d = len(selected_funds)

st.sidebar.divider()
st.sidebar.subheader("Risk parameters")

tau = st.sidebar.selectbox(
    "Horizon τ", options=[63, 126, 252, 504, 756, 1260], index=2,
    format_func=lambda t: {63: "3 months", 126: "6 months", 252: "1 year",
                           504: "2 years", 756: "3 years", 1260: "5 years"}[t])
delta = st.sidebar.selectbox(
    "Rolling step δ", options=[1, 5, 22, 66], index=2,
    format_func=lambda x: f"{eng.DELTA_LABEL[x].capitalize()}  (δ={x})")
alpha = st.sidebar.selectbox(
    "Confidence α", options=[0.90, 0.95, 0.99], index=1,
    format_func=lambda a: f"{a*100:.0f}%")

st.sidebar.divider()
n_sim       = st.sidebar.select_slider("Bootstrap draws", options=[1000, 5000, 10000, 50000], value=10000)
run_stress  = st.sidebar.checkbox("Run stress scenarios", value=True)
run_sweeps  = st.sidebar.checkbox("Run sensitivity sweeps (d, δ, τ)", value=True)

INITIAL_VALUE = eng.INITIAL_VALUE


# ══════════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════════
st.title("📉  Portfolio VaR / CVaR & Stress Testing")

if d == 0:
    st.warning("👈 Select at least one fund in the sidebar to run the analysis.")
    st.stop()

st.markdown(
    f"**{d}** equally-weighted fund(s) · capital **${INITIAL_VALUE:,.0f}** · "
    f"horizon **{tau/252:.2g} yr** · step **{eng.DELTA_LABEL[delta]}** · "
    f"confidence **{alpha*100:.0f}%**")


# ── Run the core analysis ──────────────────────────────────────────────────────
analysis = eng.run_analysis(data, selected_funds, TS, TE, tau, delta, alpha,
                            n_sim, INITIAL_VALUE)
results  = analysis["results"]
pr       = analysis["portfolio_returns"]
L        = analysis["L"]

if L < 5:
    st.error(f"Only {L} rolling windows for τ={tau}, δ={delta} over this period — "
             f"too few to bootstrap. Lower τ or δ.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Results summary  (mirrors the notebook Summary section)
# ══════════════════════════════════════════════════════════════════════════════
st.header("Results summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"VaR ({alpha*100:.0f}%)",  f"${results['VaR']:,.0f}",
          f"{-results['VaR_return']*100:.2f}% loss", delta_color="inverse")
c2.metric(f"CVaR ({alpha*100:.0f}%)", f"${results['CVaR']:,.0f}",
          f"{-results['CVaR_return']*100:.2f}% loss", delta_color="inverse")
c3.metric("Mean τ-return", f"{pr.mean()*100:+.2f}%")
c4.metric("Observations L", f"{L}")

with st.expander("Holdings & distribution detail"):
    holdings = pd.DataFrame({
        "Fund":   [labels[f] for f in selected_funds],
        "Type":   ["equity" if f in data["eq_cols"] else "bond" for f in selected_funds],
        "AUM ($B)": [latest_aum.get(f, 0.0) for f in selected_funds],
        "Weight": [f"{100/d:.1f}%"] * d,
    })
    st.dataframe(holdings, hide_index=True, width="stretch")
    st.write(
        f"Std dev **{pr.std()*100:.2f}%**  ·  "
        f"min **{pr.min()*100:+.2f}%**  ·  max **{pr.max()*100:+.2f}%**  ·  "
        f"skew **{pd.Series(pr).skew():.2f}**  ·  "
        f"excess kurtosis **{pd.Series(pr).kurtosis():.2f}**")


# ══════════════════════════════════════════════════════════════════════════════
#  Return distribution  (histogram + empirical CDF)
# ══════════════════════════════════════════════════════════════════════════════
st.header("Simulated return distribution")

sim    = results["sim"]
var_r  = results["VaR_return"]
cvar_r = results["CVaR_return"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

n, edges = np.histogram(sim, bins=60)
centers  = (edges[:-1] + edges[1:]) / 2
width    = edges[1] - edges[0]
colors   = np.where(centers <= cvar_r, "#c0392b",
           np.where(centers <= var_r,  "#e67e22", "#2980b9"))
ax1.bar(centers, n, width=width, color=colors, alpha=0.85, edgecolor="white", linewidth=0.2)
ax1.axvline(var_r,  color="#e67e22", lw=2, ls="--")
ax1.axvline(cvar_r, color="#c0392b", lw=2, ls="--")
ax1.legend(handles=[
    mpatches.Patch(color="#2980b9", label="Return > VaR threshold"),
    mpatches.Patch(color="#e67e22", label="CVaR < return ≤ VaR"),
    mpatches.Patch(color="#c0392b", label="Return ≤ CVaR (deep tail)"),
    plt.Line2D([0], [0], color="#e67e22", lw=2, ls="--", label=f"VaR  = ${results['VaR']:,.0f}"),
    plt.Line2D([0], [0], color="#c0392b", lw=2, ls="--", label=f"CVaR = ${results['CVaR']:,.0f}"),
], fontsize=8, framealpha=0.9)
ax1.xaxis.set_major_formatter(fmt_pct)
ax1.set_xlabel("Portfolio τ-horizon return"); ax1.set_ylabel("Frequency")
ax1.set_title("Return distribution")

sorted_r = np.sort(sim)
cdf      = np.linspace(1 / n_sim, 1, n_sim)
ax2.plot(sorted_r, cdf, color="#2c3e50", lw=1.5)
ax2.axvline(var_r,  color="#e67e22", lw=2, ls="--")
ax2.axvline(cvar_r, color="#c0392b", lw=2, ls="--")
ax2.axhline(1 - alpha, color="gray", lw=1, ls=":")
ax2.fill_betweenx([0, 1 - alpha], sorted_r.min(), var_r, alpha=0.08, color="#e74c3c")
ax2.xaxis.set_major_formatter(fmt_pct)
ax2.set_xlabel("Portfolio τ-horizon return"); ax2.set_ylabel("Cumulative probability")
ax2.set_title("Empirical CDF"); ax2.set_ylim(0, 1)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Rolling return time series
fig2, ax = plt.subplots(figsize=(14, 4))
rs = pd.Series(pr, index=analysis["start_dates"])
ax.plot(rs.index, rs.values, lw=0.8, color="#2c3e50", alpha=0.8)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.fill_between(rs.index, rs.values, 0, where=rs.values < 0, color="#e74c3c", alpha=0.3)
ax.fill_between(rs.index, rs.values, 0, where=rs.values >= 0, color="#27ae60", alpha=0.2)
ax.yaxis.set_major_formatter(fmt_pct)
ax.set_xlabel("Window start date"); ax.set_ylabel(f"{tau}-day return")
ax.set_title("Rolling portfolio returns")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  Sensitivity sweeps
# ══════════════════════════════════════════════════════════════════════════════
if run_sweeps:
    st.header("Sensitivity analysis")
    with st.spinner("Running sweeps …"):
        d_vals, vars_d, cvars_d = eng.sweep_d(data, fund_pool, TS, TE, tau, delta, alpha, n_sim, INITIAL_VALUE)
        dl_sweep, vars_dl, cvars_dl, Ls = eng.sweep_delta(data, selected_funds, TS, TE, tau, alpha, n_sim, INITIAL_VALUE)
        t_sweep, t_labels, vars_t, cvars_t = eng.sweep_tau(data, selected_funds, TS, TE, delta, alpha, n_sim, INITIAL_VALUE)

    cda, cdb = st.columns(2)

    with cda:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(d_vals, vars_d,  "o-", color="#e67e22", lw=2, ms=5, label="VaR")
        ax.plot(d_vals, cvars_d, "s-", color="#c0392b", lw=2, ms=5, label="CVaR")
        ax.fill_between(d_vals, vars_d, cvars_d, alpha=0.10, color="#e74c3c")
        ax.axvline(d, color="gray", lw=1.2, ls=":", label=f"Current d={d}")
        ax.yaxis.set_major_formatter(fmt_dollar)
        ax.set_xlabel("Number of funds  d"); ax.set_ylabel("Risk ($)")
        ax.set_title("Diversification: VaR & CVaR vs. d\n(random portfolios from the pool)")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with cdb:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_labels, vars_t,  "o-", color="#e67e22", lw=2, ms=7, label="VaR")
        ax.plot(t_labels, cvars_t, "s-", color="#c0392b", lw=2, ms=7, label="CVaR")
        ax.fill_between(t_labels, vars_t, cvars_t, alpha=0.10, color="#e74c3c")
        ax.yaxis.set_major_formatter(fmt_dollar)
        ax.set_xlabel("Investment horizon  τ"); ax.set_ylabel("Risk ($)")
        ax.set_title("VaR & CVaR vs. horizon τ")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4))
    x, w = np.arange(4), 0.35
    dl_labels = ["Daily\n(δ=1)", "Weekly\n(δ=5)", "Monthly\n(δ=22)", "Quarterly\n(δ=66)"]
    axa.bar(x - w/2, vars_dl,  w, color="#e67e22", alpha=0.85, label="VaR")
    axa.bar(x + w/2, cvars_dl, w, color="#c0392b", alpha=0.85, label="CVaR")
    axa.set_xticks(x); axa.set_xticklabels(dl_labels)
    axa.yaxis.set_major_formatter(fmt_dollar)
    axa.set_title("VaR & CVaR vs. rolling step δ"); axa.legend()
    axb.bar(dl_labels, Ls, color="#2980b9", alpha=0.85)
    for i, v in enumerate(Ls):
        axb.text(i, v + 2, str(v), ha="center", fontsize=9)
    axb.set_ylabel("Observations L"); axb.set_title("Sample size L vs. δ")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Stress testing
# ══════════════════════════════════════════════════════════════════════════════
if run_stress:
    st.header("Stress testing")
    stress_inputs = eng.prepare_stress_inputs(data, selected_funds)
    sr = eng.run_stress(analysis["fund_returns"], selected_funds, stress_inputs,
                        results["VaR"], results["CVaR"], alpha, n_sim, INITIAL_VALUE)

    table = pd.DataFrame([{
        "Scenario":   r["label"].replace("\n", " "),
        "VaR ($)":    r["VaR"],
        "CVaR ($)":   r["CVaR"],
        "ΔVaR ($)":   r["dVaR"],
        "ΔCVaR ($)":  r["dCVaR"],
    } for r in sr])
    st.dataframe(
        table.style.format({"VaR ($)": "${:,.0f}", "CVaR ($)": "${:,.0f}",
                            "ΔVaR ($)": "${:+,.0f}", "ΔCVaR ($)": "${:+,.0f}"}),
        hide_index=True, width="stretch")

    s_labels  = ["Baseline"] + [r["label"] for r in sr]
    all_vars  = [results["VaR"]]  + [r["VaR"]  for r in sr]
    all_cvars = [results["CVaR"]] + [r["CVaR"] for r in sr]
    x, w = np.arange(len(s_labels)), 0.38

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 5))
    axl.bar(x - w/2, all_vars,  w, color="#e67e22", alpha=0.88, label="VaR")
    axl.bar(x + w/2, all_cvars, w, color="#c0392b", alpha=0.88, label="CVaR")
    axl.axhline(results["VaR"],  color="#e67e22", lw=1, ls=":", alpha=0.6)
    axl.axhline(results["CVaR"], color="#c0392b", lw=1, ls=":", alpha=0.6)
    axl.set_xticks(x); axl.set_xticklabels(s_labels, rotation=18, ha="right", fontsize=8)
    axl.yaxis.set_major_formatter(fmt_dollar)
    axl.set_ylabel("Risk ($)"); axl.set_title("Stressed VaR & CVaR"); axl.legend()

    xs = np.arange(len(sr))
    axr.bar(xs - w/2, [r["dVaR"]  for r in sr], w, color="#e67e22", alpha=0.88, label="ΔVaR")
    axr.bar(xs + w/2, [r["dCVaR"] for r in sr], w, color="#c0392b", alpha=0.88, label="ΔCVaR")
    axr.axhline(0, color="black", lw=0.8)
    axr.set_xticks(xs); axr.set_xticklabels([r["label"].replace("\n", " ") for r in sr],
                                            rotation=18, ha="right", fontsize=8)
    axr.yaxis.set_major_formatter(fmt_dollar)
    axr.set_ylabel("Change vs. baseline ($)"); axr.set_title("Incremental risk increase"); axr.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    worst = max(sr, key=lambda r: r["dVaR"])
    st.info(f"**Worst scenario:** {worst['label'].replace(chr(10), ' ')} — "
            f"stressed VaR **${worst['VaR']:,.0f}** (Δ ${worst['dVaR']:+,.0f}), "
            f"stressed CVaR **${worst['CVaR']:,.0f}** (Δ ${worst['dCVaR']:+,.0f}).")

st.caption("Built on the IEOR 4703 term-project engine · equity → GBM/bridge, "
           "bond → duration-carry imputation.")