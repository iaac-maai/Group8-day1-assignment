---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - sklearn
  - tabular-regression
  - built-environment
  - housing
  - iaac-maai
  - ai-methods-1
metrics:
  - mae
  - rmse
  - r2
---

# Housing affordability forecast: next year's affordability from this year's indicators

This model predicts **next year's housing affordability** (house prices relative to household income, as an index) for a country, from the housing and income indicators known **this year**. It is a scikit-learn pipeline (preprocessing + Ridge regression) trained on 26 countries, 1977–2014. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/Group8-day1-assignment
- **Team:** Group 8: Akshat Dayal, Bhavini, Bo

## Intended use

- **Intended users and use:** housing policy analysts, students and researchers who want a quick **one-year-ahead** estimate of affordability per country, as an early warning before official statistics are published, or to explore what-if scenarios (e.g. faster price growth).
- **Out-of-scope uses:**
  - countries outside the 26 in the training data
  - single cities or regions inside a country
  - forecasts more than one year ahead
  - comparing affordability **levels** between countries (the inputs are indexes with country-specific base years)
  - any decision about individual households or properties

## Model details

| | |
|---|---|
| **Problem type** | Regression |
| **Algorithm** | `Ridge` (linear regression with L2 regularisation) |
| **Key hyperparameters** | `alpha=1.0` (chosen with `GridSearchCV` over 0.01–100) |
| **Target** | `log10(affordability_next)`: the model predicts the **log10** of next year's affordability |
| **Preprocessing** | Numeric: median imputation + standard scaling. `region`: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`). `decade`: ordinal encoding with the order 1970s < 1980s < … < 2020s |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` (preprocessing + model) |
| **Library versions** | scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6, joblib 1.5.3, Python 3.14 |

## Input features

The model expects a `pandas.DataFrame` with exactly these 17 columns (one row = one country in one year):

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `year` | numeric | year | Year of the indicators (the forecast is for `year + 1`) | median imputation, scaling |
| `affordability` | numeric | index | Real house price index ÷ real disposable income index × 100 (higher = less affordable) | median imputation, scaling |
| `log_affordability` | numeric | log10 | `log10(affordability)` | median imputation, scaling |
| `real_house_price_index` | numeric | index | House prices adjusted for inflation | median imputation, scaling |
| `real_personal_disposable_income` | numeric | index | Household income after tax, adjusted for inflation | median imputation, scaling |
| `real_hpi_change_pct` | numeric | % | Change in real house prices vs the previous year | median imputation, scaling |
| `real_income_change_pct` | numeric | % | Change in real income vs the previous year | median imputation, scaling |
| `affordability_lag1` | numeric | index | Affordability one year earlier | median imputation, scaling |
| `affordability_lag2` | numeric | index | Affordability two years earlier | median imputation, scaling |
| `affordability_3yr_avg` | numeric | index | Average affordability of the last 3 years | median imputation, scaling |
| `nominal_zero` | 0/1 | – | Original nominal price or income was 0 (hyperinflation artefact). Always 0 in the training data | median imputation, scaling |
| `hyperinflation_country` | 0/1 | – | Israel, Croatia or Slovenia | median imputation, scaling |
| `price_shock_up` | 0/1 | – | Real prices rose more than 30% this year | median imputation, scaling |
| `price_shock_down` | 0/1 | – | Real prices fell more than 30% this year. Always 0 in the training data | median imputation, scaling |
| `gfc_period` | 0/1 | – | Global financial crisis and aftermath (2008–2012) | median imputation, scaling |
| `region` | category | – | One of: Africa, Asia, Central/Eastern Europe, Latin America, Middle East, Nordic, North America, Oceania, Southern Europe, Western Europe | one-hot encoding |
| `decade` | category | – | `"1970s"`, `"1980s"`, … `"2020s"` | ordinal encoding |

## Output

`model.predict(X)` returns **log10 of next year's affordability**. Convert it back with `10 ** prediction`. Example: a prediction of 2.0066 means an affordability index of ≈ 101.5 next year.

## Training data

| | |
|---|---|
| **Dataset** | Housing affordability, 26 countries, 1975–2025 |
| **Source** | [Kaggle](https://www.kaggle.com/code/tolstoyjustin/forecasting-housing-affordability-to-2035/input) (column names match the Dallas Fed International House Price Database) |
| **Licence** | See the Kaggle dataset page |
| **One row is** | One country in one year |
| **Size used** | 1,210 rows: train 950 (1977–2014), test 260 (2015–2024) |
| **Coverage** | 26 countries (mostly high-income), national averages, 1977–2024 |

**Cleaning:** done on Day 1 (see the GitHub repository): rows distorted by hyperinflation removed (zero nominal values, and Croatia and Slovenia before 1995), lags and changes engineered, two leaking features removed (they used each country's full-history average, minimum and maximum), rows without lags or target dropped.

## Training procedure

- **Split:** by **time**, not at random: train 1977–2014, test 2015–2024. Neighbouring years of a country are almost identical, so a random split would leak the future.
- **Validation:** 5-fold `TimeSeriesSplit` on the training set (each fold trains on earlier years and validates on later years). Shuffled folds looked ≈ 13% better than time-based folds, which is why they were not used.
- **Models compared** (same pipeline, same folds; MAE on log10, ≈ typical % error in brackets):

| Model | CV MAE (mean ± std) |
|---|---|
| Mean baseline (`DummyRegressor`) | 0.1201 ± 0.0361 (≈ 31.9%) |
| "Next year = this year" baseline | 0.0204 ± 0.0033 (≈ 4.8%) |
| KNN (k=10) | 0.0441 ± 0.0049 (≈ 10.7%) |
| Decision tree (depth 5) | 0.0279 ± 0.0093 (≈ 6.6%) |
| Gradient boosting | 0.0235 ± 0.0080 (≈ 5.6%) |
| Random forest | 0.0204 ± 0.0055 (≈ 4.8%) |
| Linear regression | 0.0171 ± 0.0037 (≈ 4.0%) |
| **Ridge** | **0.0167 ± 0.0031 (≈ 3.9%)** |

- **Tuning:** `GridSearchCV` over `alpha` ∈ {0.01, 0.1, 1, 10, 100}. Values ≤ 1 performed the same; `alpha=1` was best.
- **Why this model:** the only model family clearly better than the "next year = this year" baseline. Tree-based models cannot predict levels outside the training range, while affordability keeps drifting. Ridge is also the fastest and fully interpretable.

## Evaluation

Results on the held-out **test set** (2015–2024, used once):

| Metric | Test (Ridge) | Test ("next year = this year") | Cross-validation (Ridge) |
|---|---|---|---|
| MAE | **3.28** index points | 3.74 | ≈ 3.9% (0.0167 log10 ± 0.0031) |
| RMSE | **4.75** index points | 5.26 | – |
| R² (log scale) | **0.83** | – | – |
| Within ±5% of the actual value | **83%** of country-years | – | – |

**Where it performs worst:** countries with sharp turns in affordability: Canada (median error 4.8%), Sweden (4.1%), Japan (4.1%), with single-year errors up to ≈ 15%. Best: Spain (0.8%), Belgium (0.9%), France (1.0%). The test error is lower than the CV error because 2015–2024 was calmer than the validation years, which include the 1980s–90s and 2008.

## Limitations and biases

- **Indexes, not money:** each country's index has its own base year, so affordability levels cannot be compared between countries, only changes over time.
- **Missing drivers:** no interest rates, mortgage rules, housing supply or migration. The model cannot foresee turns caused by them (e.g. the 2022 interest-rate rises).
- **Mostly momentum:** the model relies mainly on this year's affordability and this year's real house price growth. It extrapolates recent trends and will be late at turning points.
- **Coverage:** national averages (city-level differences are hidden); 26 mostly high-income countries; 2025 data may be estimated.
- **Small dataset:** 950 training rows. Differences of a few thousandths in MAE between models are within noise.
- **Not causal:** coefficients and importances show what the model uses, not what causes affordability.

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day2-group8-affordability", filename="model.joblib")
model = joblib.load(path)

# Spain, indicators for 2024 -> forecast for 2025
example = pd.DataFrame([{
    "year": 2024, "affordability": 100.0, "log_affordability": 2.0,
    "real_house_price_index": 100.0, "real_personal_disposable_income": 100.0,
    "real_hpi_change_pct": 2.59, "real_income_change_pct": 2.28,
    "affordability_lag1": 99.7, "affordability_lag2": 102.12, "affordability_3yr_avg": 100.61,
    "nominal_zero": 0, "hyperinflation_country": 0, "price_shock_up": 0, "price_shock_down": 0,
    "gfc_period": 0, "region": "Southern Europe", "decade": "2020s",
}])
print(10 ** model.predict(example))   # ≈ 101.5
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions of scikit-learn.

## Reproducibility

The full pipeline (data loading, training, evaluation) is in the GitHub repository: https://github.com/iaac-maai/Group8-day1-assignment. Run `notebook/Day2_models.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Group 8: Akshat Dayal, Bhavini, Bo. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is subject to its own licence (see the Kaggle dataset page).
