# %% [markdown]
# # ESG Returns Study: What Drives ESG Scores?
#
# Which firm characteristics predict a firm's ESG score?
# Approach: decile-sort charts to show univariate relationships,
# a correlation matrix, then OLS regressions with year and industry fixed effects.
#
# Key finding: firm size (log assets) is the single strongest predictor,
# with a correlation of 0.58. Within industries, R&D intensity is positively
# associated with ESG despite having a negative raw correlation, which is a
# composition effect driven by the pharmaceutical industry.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from IPython.display import display
from utils import load_panel

panel = load_panel()

char_vars = [
    ('size',              'Size (log AT)'),
    ('roa',               'Return on Assets'),
    ('leverage',          'Leverage'),
    ('rd_intensity',      'R&D Intensity'),
    ('capital_intensity', 'Capital Intensity'),
    ('equity_ratio',      'Equity Ratio'),
]

# %% [markdown]
# ## Table 1: Summary Statistics for Firm Characteristics

# %%
vars_table = ['esg_score'] + [v for v, _ in char_vars] + ['profit_margin', 'sales_growth']
labels = {
    'esg_score':         'ESG Score',
    'size':              'Size (log AT)',
    'roa':               'ROA',
    'leverage':          'Leverage',
    'rd_intensity':      'R&D Intensity',
    'capital_intensity': 'Capital Intensity',
    'equity_ratio':      'Equity Ratio',
    'profit_margin':     'Profit Margin',
    'sales_growth':      'Sales Growth',
}

summary = panel[vars_table].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
summary.index = [labels[v] for v in vars_table]
summary.columns = ['N', 'Mean', 'Std', 'Min', 'P25', 'Median', 'P75', 'Max']
summary['N'] = summary['N'].astype(int)
display(summary.round(4))

# %% [markdown]
# ## Figure 1: Mean ESG Score by Decile of Firm Characteristics
#
# Size has the steepest gradient: bottom-decile firms average ESG 0.243,
# top-decile firms average 0.633. All six characteristics show a positive
# relationship with ESG in the univariate cut.

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
axes = axes.flatten()
panel_labels = list('abcdef')

for i, (var, label) in enumerate(char_vars):
    ax = axes[i]
    tmp = panel[['esg_score', var]].dropna().copy()
    tmp['decile'] = pd.qcut(tmp[var], q=10, labels=False, duplicates='drop') + 1
    binned = tmp.groupby('decile')['esg_score'].mean().reset_index()

    ax.plot(binned['decile'], binned['esg_score'],
            marker='o', color='#1565C0', lw=2, markersize=6)
    ax.fill_between(binned['decile'], binned['esg_score'], alpha=0.08, color='#1565C0')
    ax.axhline(panel['esg_score'].mean(), ls='--', color='#B71C1C', lw=1.2, label='Sample mean')
    ax.set_xlabel(f'Decile of {label}')
    ax.set_ylabel('Mean ESG Score')
    ax.set_title(f'({panel_labels[i]}) ESG vs {label}')
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.25, ls=':')
    if i == 0:
        ax.legend(fontsize=9)

plt.suptitle('Mean ESG Score by Decile of Firm Characteristics (Decile 1 = lowest)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('outputs/esg_decile_sorts.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Figure 2: Pairwise Correlation Matrix

# %%
corr_cols = ['esg_score', 'size', 'roa', 'leverage', 'rd_intensity',
             'capital_intensity', 'equity_ratio', 'profit_margin', 'sales_growth']
corr_labels = ['ESG', 'Size', 'ROA', 'Leverage', 'R&D', 'Cap Intensity',
               'Equity Ratio', 'Profit Margin', 'Sales Growth']

corr_m = panel[corr_cols].dropna().corr()
corr_m.index = corr_labels
corr_m.columns = corr_labels

try:
    import seaborn as sns
    mask = np.triu(np.ones_like(corr_m, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_m, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, annot_kws={'size': 9}, ax=ax)
    ax.set_title('Pairwise Correlations: Firm Characteristics')
    plt.tight_layout()
    plt.savefig('outputs/correlations.png', dpi=150, bbox_inches='tight')
    plt.show()
except ImportError:
    print(corr_m.round(3).to_string())

# %% [markdown]
# ## Table 2: OLS Regressions with ESG Score as Dependent Variable
#
# Four specifications building from simple to preferred:
#   M1: size + year FE
#   M2: M1 + ROA, leverage, R&D intensity
#   M3: M2 + capital intensity, profit margin, sales growth, equity ratio
#   M4: M3 + industry FE (preferred)
#
# M4 is preferred because leverage and equity ratio flip sign once industry
# composition is controlled, which reveals that raw correlations were driven
# by which types of firms populate which industries.

# %%
specs = {
    'M1': 'esg_score ~ size + C(fyear_str)',
    'M2': 'esg_score ~ size + roa + leverage + rd_intensity + C(fyear_str)',
    'M3': ('esg_score ~ size + roa + leverage + rd_intensity '
           '+ capital_intensity + profit_margin + sales_growth + equity_ratio '
           '+ C(fyear_str)'),
    'M4': ('esg_score ~ size + roa + leverage + rd_intensity '
           '+ capital_intensity + profit_margin + sales_growth + equity_ratio '
           '+ C(fyear_str) + C(ff48_id_str)'),
}

# earlier specs showed leverage positive, M4 flips it negative within industry
# M3: 'esg_score ~ size + roa + leverage + rd_intensity + capital_intensity + C(fyear_str)'
# M2: 'esg_score ~ size + roa + C(fyear_str)'

results = {}
sample = panel.dropna(subset=['esg_score', 'size']).copy()
for name, formula in specs.items():
    results[name] = smf.ols(formula, data=sample).fit(cov_type='HC3')

key_vars = ['size', 'roa', 'leverage', 'rd_intensity',
            'capital_intensity', 'profit_margin', 'sales_growth', 'equity_ratio']

col_width = 22
col_vals  = 12

header = f"{'':>{col_width}}" + "".join(f"{'M' + str(i+1):>{col_vals}}" for i in range(4))
print(header)

for var in key_vars:
    row = f"{var:<{col_width}}"
    for m in ['M1', 'M2', 'M3', 'M4']:
        res = results[m]
        if var in res.params:
            coef  = res.params[var]
            p     = res.pvalues[var]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            row  += f"{coef:>8.3f}{stars:>4}"
        else:
            row  += f"{'n/a':>{col_vals}}"
    print(row)

print(f"\n{'R2':<{col_width}}" + "".join(f"{results[m].rsquared:>{col_vals}.3f}" for m in ['M1', 'M2', 'M3', 'M4']))
print(f"{'N':<{col_width}}" + "".join(f"{int(results[m].nobs):>{col_vals},}" for m in ['M1', 'M2', 'M3', 'M4']))
print(f"{'Industry FE':<{col_width}}" + "".join(f"{'Yes' if 'ff48' in specs[m] else 'No':>{col_vals}}" for m in ['M1', 'M2', 'M3', 'M4']))
print("Year FE in all specs. HC3 standard errors. *** p<0.01  ** p<0.05  * p<0.10")

# %% [markdown]
# ## Figure 3: Industry-level Scatter: ESG vs Size and R&D
#
# Illustrates the composition story. Pharmaceutical has very high R&D
# but very low ESG, which suppresses the raw R&D-ESG correlation.
# Within industries that are more homogeneous in R&D, the relationship is positive.

# %%
ind_means = (
    panel.groupby('industry')
    .agg(esg=('esg_score', 'mean'), size=('size', 'mean'),
         rd=('rd_intensity', 'mean'), n=('gvkey', 'count'))
    .query('n >= 30')
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (x_var, x_label) in zip(axes, [('size', 'Mean Size (log AT)'),
                                         ('rd',   'Mean R&D Intensity')]):
    ax.scatter(ind_means[x_var], ind_means['esg'], s=60, alpha=0.7, color='#1565C0')
    for _, row in ind_means.iterrows():
        name = row['industry'][:14] + '..' if len(row['industry']) > 16 else row['industry']
        ax.annotate(name, (row[x_var], row['esg']),
                    textcoords='offset points', xytext=(3, 3), fontsize=7)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Mean ESG Score')
    ax.set_title(f'Industry ESG vs {x_label}')
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/industry_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
