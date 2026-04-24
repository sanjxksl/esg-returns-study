# %% [markdown]
# # ESG Returns Study: Double Materiality by Industry
#
# Double materiality means ESG is material both financially (it predicts profits)
# and in terms of real-world impact (high-ESG firms actually score well on
# sustainability criteria). This script classifies 44 FF48 industries into four
# archetypes using cross-industry median ESG and future ROA as thresholds:
#
# Double winner: high ESG, high future ROA (13 industries)
# High ESG / Low ROA: sustainability premium, low profit (9 industries)
# Low ESG / High ROA: sin stocks or neglected assets (9 industries)
# Double laggard: low ESG, low future ROA (13 industries)
#
# Best industry for combined ESG and profitability: Measuring Instruments (FF48 36)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
from IPython.display import display
from utils import load_panel

warnings.filterwarnings('ignore')
panel = load_panel()

# %% [markdown]
# ## Table 1: Industry-Level Summary Statistics
#
# Industries with fewer than 30 observations are excluded as too noisy to interpret.

# %%
ind_stats = (
    panel
    .groupby(['ff48_id', 'industry'])
    .agg(
        n_obs            =('gvkey',             'count'),
        n_firms          =('gvkey',             'nunique'),
        mean_esg         =('esg_score',         'mean'),
        median_esg       =('esg_score',         'median'),
        mean_future_roa  =('future_roa',        'mean'),
        median_future_roa=('future_roa',        'median'),
        mean_ret         =('future_annual_ret',  'mean'),
        median_ret       =('future_annual_ret',  'median'),
        mean_roa         =('roa',               'mean'),
        mean_size        =('size',              'mean'),
        mean_leverage    =('leverage',          'mean'),
        mean_rd          =('rd_intensity',      'mean'),
    )
    .reset_index()
    .query('n_obs >= 30')
    .copy()
)

print(f"Industries with 30 or more observations: {len(ind_stats)}")

# use cross-industry medians as thresholds for the materiality matrix
esg_med = ind_stats['mean_esg'].median()
roa_med = ind_stats['mean_future_roa'].median()
ret_med = ind_stats['mean_ret'].median()


def classify_quadrant(row):
    hi_esg = row['mean_esg'] >= esg_med
    hi_roa = row['mean_future_roa'] >= roa_med
    if hi_esg and hi_roa:
        return 'Double winner'
    elif hi_esg and not hi_roa:
        return 'High ESG / Low ROA'
    elif not hi_esg and hi_roa:
        return 'Low ESG / High ROA'
    else:
        return 'Double laggard'

# earlier iteration used absolute ESG cutoff of 0.40, kept for reference
# def classify_quadrant_v1(row):
#     hi_esg = row['mean_esg'] >= 0.40
#     hi_roa = row['mean_future_roa'] >= roa_med
#     ...

ind_stats['quadrant'] = ind_stats.apply(classify_quadrant, axis=1)

display(
    ind_stats[['industry', 'n_firms', 'mean_esg', 'mean_future_roa', 'mean_ret', 'quadrant']]
    .sort_values('mean_esg', ascending=False)
    .rename(columns={'industry':'Industry', 'n_firms':'Firms',
                     'mean_esg':'ESG', 'mean_future_roa':'Fut ROA',
                     'mean_ret':'Fut Ret', 'quadrant':'Quadrant'})
    .reset_index(drop=True)
)

# %% [markdown]
# ## Figure 1: Double Materiality Matrix (ESG vs Future ROA)

# %%
COLORS = {
    'Double winner':      '#1a7744',
    'High ESG / Low ROA': '#e67e22',
    'Low ESG / High ROA': '#2980b9',
    'Double laggard':     '#c0392b',
}

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

for quad, grp in ind_stats.groupby('quadrant'):
    sizes = (grp['n_firms'] / ind_stats['n_firms'].max()) * 1800 + 80
    ax.scatter(grp['mean_esg'], grp['mean_future_roa'], s=sizes,
               c=COLORS[quad], alpha=0.82, label=quad,
               edgecolors='white', linewidths=0.8, zorder=3)

for _, row in ind_stats.iterrows():
    name = row['industry'][:16] + '..' if len(row['industry']) > 18 else row['industry']
    ax.annotate(name, (row['mean_esg'], row['mean_future_roa']),
                xytext=(4, 4), textcoords='offset points', fontsize=7.2, alpha=0.88)

ax.axvline(esg_med, color='#7f8c8d', lw=1.4, ls='--', alpha=0.7, label='Median ESG')
ax.axhline(roa_med, color='#7f8c8d', lw=1.4, ls=':',  alpha=0.7, label='Median ROA')

xlim, ylim = ax.get_xlim(), ax.get_ylim()
kw = dict(fontsize=9, fontweight='bold', alpha=0.2, ha='center', va='center')
ax.text((esg_med + xlim[1])/2, (roa_med + ylim[1])/2, 'DOUBLE WINNER',    color='#1a7744', **kw)
ax.text((esg_med + xlim[0])/2, (roa_med + ylim[1])/2, 'LOW ESG / HIGH ROA', color='#2980b9', **kw)
ax.text((esg_med + xlim[1])/2, (roa_med + ylim[0])/2, 'HIGH ESG / LOW ROA', color='#e67e22', **kw)
ax.text((esg_med + xlim[0])/2, (roa_med + ylim[0])/2, 'DOUBLE LAGGARD',   color='#c0392b', **kw)

ax.set_xlabel('Mean ESG Score')
ax.set_ylabel('Mean Future ROA')
ax.set_title('Double Materiality Matrix: ESG vs Future ROA\n'
             'Bubble size = number of firms | FF48 | 2013-2023', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.18)
plt.tight_layout()
plt.savefig('outputs/materiality_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Quadrant counts:")
print(ind_stats['quadrant'].value_counts().to_string())

# %% [markdown]
# ## Figure 2: ESG vs Stock Returns by Industry
#
# No strong ESG-return alignment at the industry level, consistent with
# the pooled regression finding that ESG does not predict returns.

# %%
def classify_quadrant_ret(row):
    hi_esg = row['mean_esg'] >= esg_med
    hi_ret = row['mean_ret'] >= ret_med
    if hi_esg and hi_ret:
        return 'High ESG / High Return'
    elif hi_esg and not hi_ret:
        return 'High ESG / Low Return'
    elif not hi_esg and hi_ret:
        return 'Low ESG / High Return'
    else:
        return 'Low ESG / Low Return'

ind_stats['quadrant_ret'] = ind_stats.apply(classify_quadrant_ret, axis=1)

RET_COLORS = {
    'High ESG / High Return': '#1a7744',
    'High ESG / Low Return':  '#e67e22',
    'Low ESG / High Return':  '#2980b9',
    'Low ESG / Low Return':   '#c0392b',
}

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

for quad, grp in ind_stats.groupby('quadrant_ret'):
    sizes = (grp['n_firms'] / ind_stats['n_firms'].max()) * 1800 + 80
    ax.scatter(grp['mean_esg'], grp['mean_ret'], s=sizes, c=RET_COLORS[quad],
               alpha=0.82, label=quad, edgecolors='white', linewidths=0.8, zorder=3)

for _, row in ind_stats.iterrows():
    name = row['industry'][:16] + '..' if len(row['industry']) > 18 else row['industry']
    ax.annotate(name, (row['mean_esg'], row['mean_ret']),
                xytext=(4, 4), textcoords='offset points', fontsize=7.2, alpha=0.88)

ax.axvline(esg_med, color='#7f8c8d', lw=1.4, ls='--', alpha=0.7)
ax.axhline(ret_med, color='#7f8c8d', lw=1.4, ls=':',  alpha=0.7)
ax.set_xlabel('Mean ESG Score')
ax.set_ylabel('Mean Future Annual Return')
ax.set_title('ESG Score vs Future Stock Returns by Industry (2013-2023)', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.18)
plt.tight_layout()
plt.savefig('outputs/esg_vs_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Within-Industry ESG to Future ROA Regressions
#
# For each industry with at least 50 observations, runs:
#     future_roa ~ esg_score + year fixed effects
#
# Key finding: Pharmaceutical Products has the highest within-industry
# ESG coefficient despite having the lowest mean ESG score. Within pharma,
# investing in ESG strongly predicts profitability, but the industry as a whole
# scores poorly due to drug pricing and access controversies at the sector level.

# %%
MIN_OBS = 50

# earlier iteration used MIN_OBS = 30, kept for reference
# MIN_OBS = 30

reg_rows = []
for (ff48_id, industry), grp in panel.groupby(['ff48_id', 'industry']):
    sub = grp.dropna(subset=['future_roa', 'esg_score', 'fyear_str'])
    if len(sub) < MIN_OBS:
        continue
    formula = (
        'future_roa ~ esg_score + C(fyear_str)'
        if sub['fyear_str'].nunique() >= 3
        else 'future_roa ~ esg_score'
    )
    try:
        m = smf.ols(formula, data=sub).fit(cov_type='HC3')
        reg_rows.append({
            'ff48_id':  ff48_id,
            'industry': industry,
            'n_obs':    len(sub),
            'esg_coef': m.params['esg_score'],
            'esg_se':   m.bse['esg_score'],
            'esg_p':    m.pvalues['esg_score'],
            'r2':       m.rsquared,
        })
    except Exception:
        pass

reg_results = pd.DataFrame(reg_rows)
reg_results['sig'] = reg_results['esg_p'].map(
    lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else '')

print(f"Regressions run: {len(reg_results)}")
print("\nTop 10 industries by ESG to ROA coefficient:")
display(
    reg_results.nlargest(10, 'esg_coef')
    [['industry', 'n_obs', 'esg_coef', 'esg_p', 'sig', 'r2']]
    .round(4)
    .reset_index(drop=True)
)

# %% [markdown]
# ## Figure 3: Within-Industry ESG Coefficient Plot

# %%
plot_df = reg_results.dropna(subset=['esg_coef']).sort_values('esg_coef')

fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.32)))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

bar_colors = ['#1a7744' if c > 0 else '#c0392b' for c in plot_df['esg_coef']]
ax.barh(plot_df['industry'], plot_df['esg_coef'], color=bar_colors, alpha=0.85, height=0.7)

for i, (_, row) in enumerate(plot_df.iterrows()):
    if row['sig']:
        xpos = row['esg_coef'] + (0.003 if row['esg_coef'] >= 0 else -0.003)
        ha = 'left' if row['esg_coef'] >= 0 else 'right'
        ax.text(xpos, i, row['sig'], va='center', ha=ha, fontsize=8, color='#2c3e50')

ax.axvline(0, color='#2c3e50', lw=1)
ax.set_xlabel('ESG to Future ROA coefficient (within-industry OLS, year FEs, HC3 SE)')
ax.set_title('Within-Industry ESG Financial Materiality\n'
             'Positive = higher ESG predicts higher future ROA   * p<.10  ** p<.05  *** p<.01',
             fontweight='bold')
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
plt.savefig('outputs/within_industry_coefs.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Full Materiality Table by Quadrant

# %%
ind_stats = ind_stats.merge(
    reg_results[['ff48_id', 'esg_coef', 'esg_p', 'sig']],
    on='ff48_id', how='left'
)

for label, q_val in [
    ('Double Winners: High ESG and High ROA',   'Double winner'),
    ('High ESG / Low ROA',                       'High ESG / Low ROA'),
    ('Low ESG / High ROA',                       'Low ESG / High ROA'),
    ('Double Laggards: Low ESG and Low ROA',     'Double laggard'),
]:
    subset = (
        ind_stats[ind_stats['quadrant'] == q_val]
        [['industry', 'n_firms', 'mean_esg', 'mean_future_roa', 'mean_ret', 'esg_coef', 'sig']]
        .rename(columns={'industry':'Industry', 'n_firms':'Firms', 'mean_esg':'ESG',
                         'mean_future_roa':'Fut ROA', 'mean_ret':'Fut Ret',
                         'esg_coef':'ESG to ROA', 'sig':'Sig'})
        .sort_values('ESG', ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n{label}")
    display(subset.round(4))

# %% [markdown]
# ## Recommended Industry: Measuring Instruments (FF48 36)
#
# Selection criteria: must be a double winner with a significant ESG to ROA
# coefficient at the 5% level. Among qualifying industries, pick the highest
# within-industry ESG coefficient.

# %%
candidates = ind_stats[
    (ind_stats['quadrant'] == 'Double winner') &
    (ind_stats['esg_p'] < 0.05)
].copy()

# fallback to all double winners if none pass the significance filter
if len(candidates) == 0:
    candidates = ind_stats[ind_stats['quadrant'] == 'Double winner'].copy()

best = candidates.sort_values('esg_coef', ascending=False).iloc[0]
print(f"Recommended industry: {best['industry']} (FF48 {int(best['ff48_id'])})")
print(f"Mean ESG:        {best['mean_esg']:.4f}")
print(f"Mean future ROA: {best['mean_future_roa']:.4f}")
print(f"Mean future ret: {best['mean_ret']:.4f}")
print(f"ESG to ROA coef: {best['esg_coef']:.4f} ({best['sig']})")

best_id = int(best['ff48_id'])
focal = panel[panel['ff48_id'] == best_id].copy()
print(f"\nSample: {len(focal):,} firm-years, {focal['gvkey'].nunique()} unique firms")

ts = (
    focal.groupby('fyear')
    .agg(esg=('esg_score', 'mean'), roa=('roa', 'mean'),
         future_roa=('future_roa', 'mean'), ret=('future_annual_ret', 'mean'))
)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(ts.index, ts['esg'], marker='o', color='#1565C0', lw=2, label='Mean ESG')
ax1.set_ylabel('ESG Score', color='#1565C0')
ax1.tick_params(axis='y', labelcolor='#1565C0')
ax1.set_xlabel('Fiscal Year')
ax1.set_title(f'{best["industry"]}: ESG and Financial Performance Over Time')

ax2 = ax1.twinx()
ax2.plot(ts.index, ts['future_roa'], marker='s', color='#1a7744', lw=2, ls='--', label='Future ROA')
ax2.plot(ts.index, ts['ret'],        marker='^', color='#B71C1C', lw=2, ls=':',  label='Future Return')
ax2.set_ylabel('Return / ROA')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
ax1.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('outputs/focal_industry_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
