# ESG and Financial Performance

An empirical study of how ESG (Environmental, Social, and Governance) scores relate
to firm profitability and stock returns using US public firms from 2013 to 2023.

Data: 20,758 firm-year observations from WRDS (Compustat, CRSP, Refinitiv ESG),
classified into Fama-French 48 industries.


## What we found

### 1. ESG predicts profitability but not stock returns

A 0.1-unit increase in ESG score predicts roughly +0.3 percentage points higher
future ROA after controlling for firm size, leverage, R&D, and industry/year fixed
effects. The same ESG score has no significant effect on returns once you control
for earnings news, meaning the market has already priced in the ESG-earnings link.

### 2. Firm size is the biggest driver of ESG scores

ESG-size correlation is 0.58, by far the largest pairwise correlation. Bottom-decile
firms by size average an ESG score of 0.24 while top-decile firms average 0.63.
R&D intensity has a negative raw correlation with ESG due to pharmaceutical firms,
but turns positive within industries, a classic composition effect.

### 3. Four industry archetypes

| Quadrant | ESG | Future ROA | Example industries |
|---|---|---|---|
| Double winner | High | High | Chemicals, Measuring Instruments, Retail |
| High ESG / Low ROA | High | Low | Utilities, Precious Metals |
| Low ESG / High ROA | Low | High | Trading, Coal, Construction |
| Double laggard | Low | Low | Pharma, Banking |

### 4. The pharmaceutical paradox

Pharmaceutical Products has the lowest mean ESG score (0.32) but the highest
within-industry ESG to ROA coefficient (+0.68, significant at 1%). Within pharma,
firms that invest in ESG are significantly more profitable, even though the industry
as a whole is dragged down by drug pricing and access controversies at the sector level.

### 5. Best industry for ESG and profitability: Measuring Instruments (FF48 36)

Mean ESG 0.47, mean future ROA 0.050, ESG to ROA coefficient +0.18 (significant at 1%).
335 firm-years, 54 unique firms, consistent outperformance from 2013 to 2023.


## Data

The panel data is not included in this repo. It is sourced from WRDS (Wharton Research
Data Services) which has licensing restrictions that prohibit redistribution.

To generate it, run the data_prepare.ipynb notebook from your local copy of the original
project using your own WRDS credentials. It will produce a file called
esg_financial_panel_2013_2023.csv. Place that file in a data/ folder at the project root
before running any of the analysis scripts.


## Project structure

```
esg-returns-study/
    data/                    not tracked, add the CSV here after running data_prepare
    utils/
        ff48.py              Fama-French 48 industry classification
        panel.py             load_panel() loads, cleans, and builds all variables
        __init__.py
    descriptive.py           ESG distribution, industry ranking, time trend
    determinants.py          What predicts ESG scores (size, R&D, leverage)
    financial_impact.py      Does ESG predict ROA or returns
    industry_materiality.py  Double materiality matrix and focal industry
    outputs/                 Generated figures saved here
```


## How to run

Open any script in VS Code and run cells interactively using the Jupyter extension
(click Run Cell above any # %% block), or run from the terminal:

```
python descriptive.py
python determinants.py
python financial_impact.py
python industry_materiality.py
```

All scripts import shared setup from utils/panel.py so there is no repeated
boilerplate. Run from the project root so the utils import resolves correctly.


## Dependencies

```
pandas >= 1.5
numpy
matplotlib
statsmodels
seaborn          optional, used for correlation heatmap and industry heatmap
wrds             only needed for the robustness block in financial_impact.py
```


## Ideas for further work

Panel fixed effects: use linearmodels to add firm FEs and look at within-firm ESG changes

ESG pillar breakdown: split ESG into E, S, and G sub-scores and test each separately

Portfolio analysis: construct long/short portfolios sorted on ESG within each industry

Causal identification: use an ESG regulation shock like EU SFDR 2021 as an instrument

Newer data: extend the WRDS pull to 2024-2025 to capture post-COVID ESG shifts

Sector deep-dive: replicate the focal-industry analysis for all double-winner industries
