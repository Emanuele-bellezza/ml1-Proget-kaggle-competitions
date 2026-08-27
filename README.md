# ML1 Kaggle Competitions — Salary Regression & Restaurant Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

Two supervised learning competitions from the Machine Learning 1 course at the University of Warsaw. Both use only algorithms covered in class: Linear Regression, Ridge, Lasso, ElasticNet, KNN, and SVR/SVC.

**Course:** Machine Learning 1 · University of Warsaw (Erasmus) · 2025/26  
**Author:** Emanuele Bellezza (K-18722) | [LinkedIn](https://www.linkedin.com/in/emanuele-bellezza-957704256) | [GitHub](https://github.com/Emanuele-bellezza)

---

## Projects Overview

| Task | Target | Metric | Best Model | Score |
|------|--------|--------|-----------|-------|
| [Regression](#1-regression--developer-salary-prediction) | Developer annual salary (USD) | RMSE | SVR (RBF kernel) | ~$34,090 RMSE |
| [Classification](#2-classification--restaurant-failure-prediction) | Restaurant closure (binary) | Balanced Accuracy | Logistic Regression (L2) + PowerTransformer | 0.6852 BA |

---

## 1. Regression — Developer Salary Prediction

### Problem
Predict the annual salary (`annual.pay.usd`) of software developers based on 40 features including experience, location, programming languages, and job characteristics. 
**Business Value:** Predicting salaries helps companies optimize their HR compensation strategies and allows job boards to estimate market values dynamically.

**Dataset:** 2,512 training observations · 628 test observations · 40 features

### Approach

#### Preprocessing
- **Monthly salary fix:** identified ~378 observations where pay < $10k but × 12 fell in the valid range → converted to annual
- **Outlier removal:** dropped observations outside [$10k, $500k]
- **Log transformation of target:** `log1p(annual.pay.usd)` — reduced skewness from 33 to ~1
- **Train/test split before any preprocessing** to prevent data leakage

#### Feature Engineering (346 features generated)
Key constructed features:
- `exp_bucket` — discretized experience (5 levels) · strongest predictor (corr +0.41)
- `region_target_enc` — median log-salary by region (corr +0.22)
- `senior_x_top_region` — interaction: seniority × high-wage region (corr +0.29)
- `remote_senior_top` — 3-way interaction: remote × senior × top region
- `age_squared` — captures salary peak at 35–44, then decline
- Log transforms of experience variables

#### Feature Selection
Three strategies compared:

| Approach | Features | SVR RMSE (CV) |
|----------|----------|---------------|
| No selection (raw) | 346 | ~$36,000 |
| **Statistical** (Pearson + ANOVA F-test) | **126** | **~$28,735** ✅ |
| Lasso-based | 305 | ~$30,000 |

Statistical selection wins: Pearson for continuous features, ANOVA F-statistic for binary features.

#### Models & Pipeline
All models follow: `VarianceThreshold → StandardScaler → Model`  
SVR additionally uses `PowerTransformer (Yeo-Johnson)` before scaling — symmetrizes distributions for the RBF kernel distance metric.

| Model | CV RMSE (USD) |
|-------|--------------|
| Linear Regression | ~$35,000 |
| Ridge | ~$31,000 |
| Lasso | ~$32,000 |
| ElasticNet | ~$31,500 |
| KNN | ~$33,000 |
| **SVR (RBF)** | **~$28,735** ✅ |

**Why SVR wins:** the RBF kernel captures non-linear salary patterns (e.g., exponential returns to seniority) that linear models cannot express.

**Why Yeo-Johnson helps SVR only:** SVR's RBF kernel computes Euclidean distances in feature space — it's sensitive to distributional shape. Yeo-Johnson symmetrizes skewed features, improving the kernel's distance geometry.

---

## 2. Classification — Restaurant Failure Prediction

### Problem
Predict whether a restaurant will close (`status_closed`) based on 86 features including location, rating history, competition density, and temporal review trends.
**Business Value:** Forecasting restaurant closures assists commercial real estate investors in assessing tenant risk and helps food delivery platforms optimize their onboarding strategies.

**Dataset:** 33,296 training observations · 8,325 test observations · 86 features  
**Class imbalance:** 90.2% open / 9.8% closed → handled with `class_weight='balanced'`

### Approach

#### Preprocessing
- 6-group imputation strategy (rating variables → 0, temporal → median, categorical → separate "missing" category)
- Critical fix: `weekends_only` / `workdays_only` stored as strings → converted to boolean
- Pipeline: `VarianceThreshold → StandardScaler → Model`

#### Feature Engineering (121 features)
- **Marginal POI rings:** converted cumulative density counts to differential rings (100–200m, 200–500m, 500–1000m, 1000–2000m) — breaks multicollinearity (ρ > 0.95 between raw cumulative features)
- `bayesian_rating` — attenuates ratings from low-review-count restaurants
- `review_trend` — recent momentum: `ratings_1m - (ratings_3m / 3)`
- `rating_cv` — coefficient of variation: consistency as a closure signal
- `place_age_days_sq` — non-linear age effect

#### Advanced Feature Engineering (WoE + Information Value)
- Computed Information Value (IV) for all 121 features via `OptimalBinning`
- 14 features with IV > 0.3 (strong predictors); 50 features with IV < 0.02 (dropped)
- Weight of Evidence (WoE) transformation on strong/medium IV features — linearizes non-linear relationships for Logistic Regression

#### Model Results

| Model | CV Balanced Accuracy |
|-------|---------------------|
| KNN (k=5) | 0.5175 |
| Baseline Logistic (old) | 0.6566 |
| SVC RBF (C=0.1) | 0.6760 |
| Logistic pura (no penalty) | 0.6786 |
| StandardScaler + L2 | 0.6787 |
| PowerTransformer + L1 | 0.6838 |
| PowerTransformer + ElasticNet | 0.6849 |
| **PowerTransformer + L2 (Logistic Regression)** | **0.6852** ✅ |

**Kaggle public score:** BA = 0.6852 · **Rank: 9th**

**Why PowerTransformer + L2 wins:** The combination of non-linear power transformations (Yeo-Johnson) to stabilize variance and make the data more Gaussian-like, coupled with L2 regularization (Ridge) to prevent overfitting on the highly engineered features (like marginal POI rings and WoE transformations), proved superior to complex SVMs for this specific dataset.

---

## How to Run

To reproduce the analysis and model training locally:

### 1. Clone the repository
```bash
git clone https://github.com/Emanuele-bellezza/ml1-kaggle-competitions.git
cd ml1-kaggle-competitions
```

### 2. Set up the environment
It is highly recommended to use a virtual environment to manage dependencies, particularly for libraries like `optbinning`.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Notebooks
Launch Jupyter Notebook to view and execute the pipelines:
```bash
jupyter notebook
```
- Open `Regression Problem/final_pipeline_Regression_Emanuele_Bellezza_K-18722.ipynb` for the Salary prediction pipeline.
- Open `Classification Problem/Final_Pipeline_Classification_Emanuele_Bellezza_K-18722.ipynb` for the Restaurant failure prediction pipeline.

*(Note: Data paths within the notebooks are set relative to their respective directories).*

---

## Presentation Slides

A comprehensive slide deck summarizing the business context, methodology, and results for both the regression and classification problems is available here:
[`Final project ML1_Emanuele_Bellezza_K-18722.pptx`](./Final%20project%20ML1_Emanuele_Bellezza_K-18722.pptx)

---

## Repository Structure

```
ML1-kaggle-competitions/
├── Regression Problem/
│   ├── final_pipeline_Regression_Emanuele_Bellezza_K-18722.ipynb  ← main notebook
│   ├── Data/
│   │   ├── train.csv
│   │   └── test.csv
│   └── plots/                          ← EDA and model comparison charts
│
├── Classification Problem/
│   ├── Final_Pipeline_Classification_Emanuele_Bellezza_K-18722.ipynb        ← main notebook
│   ├── Data/
│   │   ├── restaurants_test.csv
│   │   └── [derived CSVs excluded via .gitignore]
│   └── Data/*.png                      ← model evaluation plots
│
├── Final project ML1_Emanuele_Bellezza_K-18722.pptx               ← presentation slides
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Pipelines, models, cross-validation |
| scipy.stats | Statistical feature selection (Pearson, ANOVA) |
| optbinning | Information Value & Weight of Evidence |
| matplotlib, seaborn | Visualizations |

---

## Key Takeaways

- **SVR with RBF kernel** outperforms linear models on salary prediction because of non-linear experience/seniority effects. Yeo-Johnson preprocessing is beneficial specifically for kernel methods sensitive to distributional symmetry.
- **Statistical feature selection** (Pearson + ANOVA) on domain-engineered features beats raw high-dimensional inputs and Lasso-based selection — reducing features from 346 to 126 while improving RMSE.
- **WoE transformation** is a powerful preprocessing step for logistic-family classifiers on tabular data with non-linear feature-target relationships.
- **Marginal spatial rings** (differential POI counts) break the near-perfect multicollinearity of cumulative distance features, providing cleaner spatial signals.

---

## License

Academic project — University of Warsaw, Erasmus exchange 2025/26.  
Data provided by the ML1 course (Kaggle-style private competition).
