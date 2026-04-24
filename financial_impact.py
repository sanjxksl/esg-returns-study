# %% [markdown]
# # ESG Returns Study: Does ESG Affect Financial Performance?
#
# Does a firm's ESG score in year t predict profitability (ROA) or stock returns in year t+1?
#
# Key findings:
# ESG positively and significantly predicts future ROA even after controls,
# with the coefficient shrinking roughly 80% once firm characteristics are added.
# ESG does not predict returns once earnings news is included, which suggests
# the market has already priced in whatever ESG signals about future earnings.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from IPython.display import display
from utils import load_panel

panel = load_panel()
print(f"Panel: {len(panel):,} observations, {panel['gvkey'].nunique():,} firms, "
      f"{panel['fyear'].min()} to {panel['fyear'].max()}")

# shared formula pieces used across all four regressions
CONTROLS = 'size + roa + leverage + rd_intensity + capital_intensity + sales_growth'
FES = 'C(fyear_str) + C(ff48_id_str)'


def show_reg(label, res, show_vars):
    """Print a compact regression summary for a subset of variables."""
    print(f"\n{label}")
    print(f"N = {int(res.nobs):,}   R2 = {res.rsquared:.4f}")
    print(f"{'Variable':<22} {'Coef':>9} {'SE':>9} {'t':>7} {'p':>9}")
    for v in show_vars:
        if v not in res.params:
            continue
        c, se, t, p = res.params[v], res.bse[v], res.tvalues[v], res.pvalues[v]
        stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f"{v:<22} {c:>9.4f} {se:>9.4f} {t:>7.2f} {p:>9.4f} {stars}")


# %% [markdown]
# ## Regressions 1 and 2: ESG Predicting Future ROA

# %%
reg1 = smf.ols(
    f'future_roa ~ esg_score + {FES}',
    data=panel.dropna(subset=['future_roa', 'esg_score'])
).fit(cov_type='HC3')

reg2 = smf.ols(
    f'future_roa ~ esg_score + {CONTROLS} + {FES}',
    data=panel.dropna(subset=['future_roa', 'esg_score', 'size', 'roa',
                               'leverage', 'rd_intensity', 'capital_intensity', 'sales_growth'])
).fit(cov_type='HC3')

# earlier iteration without industry FE, kept for reference
# reg1_no_ind_fe = smf.ols(
#     'future_roa ~ esg_score + C(fyear_str)',
#     data=panel.dropna(subset=['future_roa', 'esg_score'])
# ).fit(cov_type='HC3')

show_reg("Regression 1: Future ROA on ESG, no controls", reg1, ['esg_score'])
print("Year FE: Yes   Industry FE: Yes   Controls: No")

show_reg("Regression 2: Future ROA on ESG with firm controls", reg2,
         ['esg_score', 'size', 'roa', 'leverage', 'rd_intensity', 'capital_intensity', 'sales_growth'])
print("Year FE: Yes   Industry FE: Yes   Controls: Yes")

# %% [markdown]
# ## Regressions 3a and 3b: ESG Predicting Future Stock Returns
#
# Regression 3a finds ESG slightly negative unconditionally.
# Regression 3b shows that coefficient collapses to insignificance once
# earnings news is added, meaning the market efficiently prices the
# ESG-earnings channel and ESG itself carries no incremental return signal.

# %%
reg3a = smf.ols(
    f'future_annual_ret ~ esg_score + {FES}',
    data=panel.dropna(subset=['future_annual_ret', 'esg_score'])
).fit(cov_type='HC3')

reg3b = smf.ols(
    f'future_annual_ret ~ esg_score + earnings_news + roa + size '
    f'+ leverage + sales_growth + rd_intensity + capital_intensity + {FES}',
    data=panel.dropna(subset=['future_annual_ret', 'esg_score', 'earnings_news',
                               'roa', 'size', 'leverage', 'sales_growth',
                               'rd_intensity', 'capital_intensity'])
).fit(cov_type='HC3')

# earlier iteration using profit_margin instead of earnings_news, kept for reference
# reg3b_v1 = smf.ols(
#     f'future_annual_ret ~ esg_score + profit_margin + roa + size '
#     f'+ leverage + sales_growth + {FES}',
#     data=panel.dropna(subset=['future_annual_ret', 'esg_score', 'profit_margin',
#                                'roa', 'size', 'leverage', 'sales_growth'])
# ).fit(cov_type='HC3')

show_reg("Regression 3a: Future Return on ESG, unconditional", reg3a, ['esg_score'])
print("Year FE: Yes   Industry FE: Yes")

show_reg("Regression 3b: Future Return on ESG with earnings news and controls", reg3b,
         ['esg_score', 'earnings_news', 'roa', 'size', 'leverage',
          'sales_growth', 'rd_intensity', 'capital_intensity'])
print("Year FE: Yes   Industry FE: Yes")

# %% [markdown]
# ## Figure 1: ESG Coefficient Across All Four Specifications

# %%
spec_labels = ['ROA\n(no controls)', 'ROA\n(+controls)',
               'Return\n(no controls)', 'Return\n(+controls)']
coefs  = [reg1.params['esg_score'], reg2.params['esg_score'],
          reg3a.params['esg_score'], reg3b.params['esg_score']]
ses    = [reg1.bse['esg_score'],    reg2.bse['esg_score'],
          reg3a.bse['esg_score'],   reg3b.bse['esg_score']]
colors = ['#1565C0', '#1565C0', '#B71C1C', '#B71C1C']

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(range(4), coefs, yerr=[1.96 * s for s in ses],
            fmt='none', color='gray', capsize=5, lw=1.5, zorder=2)
ax.scatter(range(4), coefs, s=120, color=colors, zorder=3)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(spec_labels, fontsize=10)
ax.set_ylabel('ESG Coefficient (95% CI)')
ax.set_title('ESG Effect on Future ROA (blue) and Future Return (red)\n'
             'All specs: year and industry FEs, HC3 standard errors')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig('outputs/esg_coef_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Figure 2: ESG Decile vs Future ROA and Future Return
#
# The left panel shows a clear positive gradient: higher ESG deciles have higher
# future ROA. The right panel is flat, consistent with market efficiency.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (outcome, ylabel, color) in zip(axes, [
    ('future_roa',        'Mean Future ROA',         '#1565C0'),
    ('future_annual_ret', 'Mean Future Annual Return','#B71C1C'),
]):
    tmp = panel[['esg_score', outcome]].dropna().copy()
    tmp['decile'] = pd.qcut(tmp['esg_score'], q=10, labels=False, duplicates='drop') + 1
    binned = tmp.groupby('decile')[outcome].mean()

    ax.plot(binned.index, binned.values, marker='o', color=color, lw=2, markersize=6)
    ax.axhline(binned.mean(), ls='--', color='gray', lw=1, label='Overall mean')
    ax.set_xlabel('ESG Score Decile (1 = lowest)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'ESG Decile vs {ylabel}')
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.2)
    ax.legend()

plt.suptitle('ESG and Financial Performance by ESG Decile', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('outputs/esg_decile_outcomes.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Robustness: Replace Earnings News with Quarterly SUE
#
# Requires a live WRDS connection to pull quarterly earnings data from fundq.
# Set RUN_ROBUSTNESS = True if you have WRDS access.
# The finding is unchanged: ESG is insignificant once SUE is included.

# %%
RUN_ROBUSTNESS = False

if RUN_ROBUSTNESS:
    import wrds
    db = wrds.Connection()

    q_data = db.raw_sql("""
        SELECT gvkey, datadate, rdq, fyearq, fqtr, epspxq
        FROM comp.fundq
        WHERE indfmt = 'INDL' AND datafmt = 'STD' AND consol = 'C'
          AND popsrc = 'D' AND rdq IS NOT NULL AND epspxq IS NOT NULL
          AND fyearq >= 2008
    """)

    q_data['gvkey']    = q_data['gvkey'].astype(str).str.strip().str.zfill(6)
    panel['gvkey']     = panel['gvkey'].astype(str).str.strip().str.zfill(6)
    q_data['fqtr']     = pd.to_numeric(q_data['fqtr'],   errors='coerce')
    q_data['fyearq']   = pd.to_numeric(q_data['fyearq'], errors='coerce')
    q_data['datadate'] = pd.to_datetime(q_data['datadate'])
    q_data['rdq']      = pd.to_datetime(q_data['rdq'])
    q_data = (
        q_data.dropna(subset=['fqtr', 'fyearq', 'epspxq'])
        .sort_values(['gvkey', 'fyearq', 'fqtr', 'datadate'])
        .drop_duplicates(subset=['gvkey', 'fyearq', 'fqtr'], keep='last')
    )

    # seasonal random walk surprise: eps minus same quarter one year ago
    lag_eps = q_data[['gvkey', 'fyearq', 'fqtr', 'epspxq']].copy()
    lag_eps.columns = ['gvkey', 'fyearq_next', 'fqtr', 'eps_lag4q']
    lag_eps['fyearq'] = lag_eps['fyearq_next'] + 1
    lag_eps = lag_eps.drop(columns='fyearq_next')

    q_data = q_data.merge(lag_eps, on=['gvkey', 'fyearq', 'fqtr'], how='left')
    q_data['surprise'] = q_data['epspxq'] - q_data['eps_lag4q']

    # sum quarterly surprises within each firm-year to get annual SUE
    annual_sue = (
        q_data[q_data['surprise'].notna()]
        .groupby(['gvkey', 'fyearq'])['surprise']
        .sum()
        .reset_index()
        .rename(columns={'fyearq': 'fyear', 'surprise': 'sue'})
    )

    panel_sue = panel.merge(annual_sue, on=['gvkey', 'fyear'], how='left')

    reg_rob = smf.ols(
        f'future_annual_ret ~ esg_score + sue + roa + size '
        f'+ leverage + sales_growth + rd_intensity + capital_intensity + {FES}',
        data=panel_sue.dropna(subset=['future_annual_ret', 'esg_score', 'sue', 'roa',
                                       'size', 'leverage', 'sales_growth',
                                       'rd_intensity', 'capital_intensity'])
    ).fit(cov_type='HC3')

    show_reg("Robustness: Future Return on ESG with SUE and controls", reg_rob,
             ['esg_score', 'sue'])
else:
    print("Robustness block skipped. Set RUN_ROBUSTNESS = True with WRDS access.")
