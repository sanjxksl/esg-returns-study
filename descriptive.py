# %% [markdown]
# # ESG Returns Study: Sample Description
#
# Covers who is in the sample, how ESG scores are distributed across industries,
# and how ESG has changed from 2013 to 2023.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.display import display
from utils import load_panel

panel = load_panel()
print(f"Observations: {len(panel):,}")
print(f"Unique firms: {panel['gvkey'].nunique():,}")
print(f"Years: {panel['fyear'].min()} to {panel['fyear'].max()}")
print(f"Industries: {panel['industry'].nunique()}")

# %% [markdown]
# ## Table 1: Summary Statistics

# %%
stat_cols = ['esg_score', 'roa', 'future_roa', 'future_annual_ret',
             'at', 'size', 'lag_roa', 'lag_size']
stat_labels = {
    'esg_score':        'ESG Score',
    'roa':              'ROA',
    'future_roa':       'Future ROA',
    'future_annual_ret':'Future Annual Return',
    'at':               'Total Assets (M)',
    'size':             'Size (log AT)',
    'lag_roa':          'Lagged ROA',
    'lag_size':         'Lagged Size',
}

summary = panel[[c for c in stat_cols if c in panel.columns]].describe().T
summary = summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
summary.index = [stat_labels.get(v, v) for v in summary.index]
summary.columns = ['N', 'Mean', 'Std', 'Min', 'P25', 'Median', 'P75', 'Max']
summary['N'] = summary['N'].astype(int)
display(summary.round(4))

# %% [markdown]
# ## Figure 1: Mean ESG Score by Industry
#
# Only industries with at least 50 observations included to avoid noisy estimates.

# %%
esg_by_ind = (
    panel.groupby('industry')['esg_score']
    .agg(Mean='mean', Median='median', Std='std', N='count')
    .query('N >= 50')
    .sort_values('Mean', ascending=False)
    .round(4)
)

display(esg_by_ind)

sample_mean = esg_by_ind['Mean'].mean()

# darker blue for above-average industries
colors = ['#1565C0' if v >= sample_mean else '#90CAF9' for v in esg_by_ind['Mean']]

fig, ax = plt.subplots(figsize=(11, max(7, len(esg_by_ind) * 0.38)))
ax.barh(esg_by_ind.index, esg_by_ind['Mean'], color=colors)
ax.axvline(sample_mean, color='red', linestyle='--', linewidth=1.5,
           label=f'Sample Mean ({sample_mean:.3f})')
ax.set_xlabel('Mean ESG Score')
ax.set_title('Mean ESG Score by Industry (FF48, N >= 50, 2013-2023)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/esg_by_industry.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Figure 2: ESG Trend Over Time

# %%
esg_yr = (
    panel.groupby('fyear')['esg_score']
    .agg(Mean='mean', Median='median', Std='std', N='count')
    .round(4)
)
display(esg_yr.rename(columns={'N': 'Firms'}))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(esg_yr.index, esg_yr['Mean'],   marker='o', color='#1565C0', lw=2, label='Mean ESG')
ax1.plot(esg_yr.index, esg_yr['Median'], marker='s', color='#42A5F5', lw=2, ls='--', label='Median ESG')
ax1.fill_between(esg_yr.index,
                 esg_yr['Mean'] - esg_yr['Std'],
                 esg_yr['Mean'] + esg_yr['Std'],
                 alpha=0.1, color='#1565C0', label='+/- 1 Std Dev')
ax1.set_ylabel('ESG Score')
ax1.set_xlabel('Fiscal Year')
ax1.set_title('ESG Score Trend (2013-2023)')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.bar(esg_yr.index, esg_yr['N'], alpha=0.15, color='gray')
ax2.set_ylabel('Number of Firms', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('outputs/esg_over_time.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Figure 3: ESG Heatmap by Industry and Year
#
# Top 15 industries by mean ESG score, showing how each evolved year by year.

# %%
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

top15_inds = esg_by_ind.head(15).index.tolist()
heat_data = (
    panel[panel['industry'].isin(top15_inds)]
    .groupby(['industry', 'fyear'])['esg_score']
    .mean()
    .unstack('fyear')
)

if HAS_SEABORN:
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='Blues',
                linewidths=0.4, annot_kws={'size': 7.5}, ax=ax)
    ax.set_title('Mean ESG Score by Industry and Year (Top 15 Industries)')
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig('outputs/esg_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("Install seaborn for the heatmap: pip install seaborn")
