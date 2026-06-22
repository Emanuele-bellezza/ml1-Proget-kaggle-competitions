# =============================================================================
# FINAL PIPELINE — Restaurant Survival Classification
# Machine Learning 1 — Prof. Piotr Wójcik — a.y. 2025/2026
#
# This script:
#   1. Loads raw training and test data
#   2. Data preparation (imputation, encoding)
#   3. Feature engineering (base + advanced WoE/interactions)
#   4. Correlation pruning
#   5. LASSO feature selection
#   6. Trains best model (PowerTransformer + ElasticNet)
#   7. Saves submission.csv
#
# Input:  restaurants_train.csv, restaurants_test.csv
# Output: submission.csv
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from optbinning import OptimalBinning

# =============================================================================
# 1. LOAD DATA
# =============================================================================
train = pd.read_csv('restaurants_train.csv')
test  = pd.read_csv('restaurants_test.csv')

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")

# Save target and IDs
target   = train['status_closed'].copy()
train_id = train['restaurant_id'].copy()
test_id  = test['restaurant_id'].copy()

# =============================================================================
# 2. DATA PREPARATION
# =============================================================================
print("\n--- Step 2: Data Preparation ---")

# Merge train+test for deterministic transformations (no leakage)
train['_is_train'] = 1
test['_is_train']  = 0
test['status_closed'] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True)

# --- GROUP 0: Synthetic indicator for new restaurants (no reviews) ---
df['is_new_restaurant'] = (df['user_ratings_total'] == 0).astype(int)

# --- GROUP 1: Physiological NaNs → impute with 0 ---
group1 = [
    'rating_5', 'rating_4', 'rating_3', 'rating_2', 'rating_1',
    'rating_avg', 'rating_std',
    'ratings_num_1m_prior', 'ratings_num_3m_prior',
    'ratings_num_6m_prior', 'ratings_num_9m_prior', 'ratings_num_12m_prior',
    'lang_pl_count', 'rating_pl', 'rating_foreign', 'foreign_lang_share',
    'review_has_text_pct', 'review_length_avg', 'review_length_std',
]
df[group1] = df[group1].fillna(0)

# --- GROUP 3: price_level → OHE with "missing" category ---
df['price_level'] = df['price_level'].fillna('missing')
df['price_level'] = df['price_level'].replace({
    1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4',
    '1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4'
})
price_dummies = pd.get_dummies(df['price_level'], prefix='price_level', dtype=int)
df = pd.concat([df, price_dummies], axis=1)
df.drop(columns=['price_level'], inplace=True)

# --- GROUP 4: tagcat_* → impute with 0 ---
group4 = [
    'tagcat_services', 'tagcat_amenities', 'tagcat_atmosphere',
    'tagcat_offerings', 'tagcat_social_inclusivity', 'tagcat_payment_options',
]
df[group4] = df[group4].fillna(0)

# --- GROUP 5: catch_restaurant_count_* → impute with 0 ---
group5 = [
    'catch_restaurant_count_500m',
    'catch_restaurant_count_1000m',
    'catch_restaurant_count_2000m',
]
df[group5] = df[group5].fillna(0)

# --- GROUP 6: category_top20 → OHE with "missing" category ---
df['category_top20'] = df['category_top20'].fillna('missing')
cat_dummies = pd.get_dummies(df['category_top20'], prefix='cat', dtype=int)
df = pd.concat([df, cat_dummies], axis=1)
df.drop(columns=['category_top20'], inplace=True)

# --- weekends_only and workdays_only: string → 0/1 ---
df['weekends_only'] = df['weekends_only'].astype(str).str.strip().replace(
    {'True': 1, 'False': 0, 'nan': np.nan}).astype(float)
df['workdays_only'] = df['workdays_only'].astype(str).str.strip().replace(
    {'True': 1, 'False': 0, 'nan': np.nan}).astype(float)

# Split back into train and test
n_train = train_id.shape[0]
train_clean = df[df['_is_train'] == 1].copy()
test_clean  = df[df['_is_train'] == 0].copy()
train_clean.drop(columns=['_is_train', 'status_closed'], inplace=True)
test_clean.drop(columns=['_is_train', 'status_closed'], inplace=True)
train_clean['status_closed'] = target.values

print(f"After preparation — Train: {train_clean.shape}, Test: {test_clean.shape}")

# =============================================================================
# 3. FEATURE ENGINEERING — BASE (8 features)
# =============================================================================
print("\n--- Step 3: Feature Engineering (base) ---")

train_fe = train_clean.drop(columns=['status_closed']).copy()
test_fe  = test_clean.copy()
df_fe = pd.concat([train_fe, test_fe], axis=0, ignore_index=True)

# 1. POI marginal rings
df_fe['poi_ring_100_200']   = df_fe['poi_count_200m']  - df_fe['poi_count_100m']
df_fe['poi_ring_200_500']   = df_fe['poi_count_500m']  - df_fe['poi_count_200m']
df_fe['poi_ring_500_1000']  = df_fe['poi_count_1000m'] - df_fe['poi_count_500m']
df_fe['poi_ring_1000_2000'] = df_fe['poi_count_2000m'] - df_fe['poi_count_1000m']

# 2. Catch restaurant count marginal rings
df_fe['catch_ring_500_1000']  = df_fe['catch_restaurant_count_1000m'] - df_fe['catch_restaurant_count_500m']
df_fe['catch_ring_1000_2000'] = df_fe['catch_restaurant_count_2000m'] - df_fe['catch_restaurant_count_1000m']

# 3. place_age_days squared (quadratic term)
df_fe['place_age_days_sq'] = df_fe['place_age_days'] ** 2

# 4. Bayesian rating (rating adjusted for number of reviews)
m = 10
global_mean = df_fe['rating_avg'].mean()
df_fe['bayesian_rating'] = (
    (df_fe['rating_avg'] * df_fe['user_ratings_total'] + global_mean * m) /
    (df_fe['user_ratings_total'] + m)
)

# 5. Review trend (recent rating vs overall rating)
df_fe['review_trend'] = df_fe['ratings_avg_3m_prior'] - df_fe['rating_avg']

# 6. Rating coefficient of variation
df_fe['rating_cv'] = np.where(
    df_fe['rating_avg'] > 0,
    df_fe['rating_std'] / df_fe['rating_avg'],
    0
)

# 7. Polarization (share of extreme reviews)
df_fe['polarization'] = np.where(
    df_fe['user_ratings_total'] > 0,
    (df_fe['rating_5'] + df_fe['rating_1']) / df_fe['user_ratings_total'],
    0
)

# 8. Weekend dependency
df_fe['weekend_dependency'] = np.where(
    (df_fe['hours_open'] > 0) & (df_fe['hours_open'].notna()),
    df_fe['hours_open_weekends'] / df_fe['hours_open'],
    np.nan
)

# Split back
train_fe_out = df_fe.iloc[:n_train].copy()
test_fe_out  = df_fe.iloc[n_train:].copy()
train_fe_out['status_closed'] = target.values

print(f"After base FE — Train: {train_fe_out.shape}, Test: {test_fe_out.shape}")

# =============================================================================
# 4. FEATURE ENGINEERING — ADVANCED (WoE + Interactions)
# =============================================================================
print("\n--- Step 4: Advanced Feature Engineering (WoE + interactions) ---")

target_arr = target.values

# --- PART 2a: WoE on top numeric variables ---
woe_candidates = [
    'user_ratings_total', 'place_age_days', 'tagcat_amenities',
    'review_length_avg', 'polarization', 'review_has_text_pct',
]

woe_models = {}
for col in woe_candidates:
    x_tr = train_fe_out[col].values
    optb = OptimalBinning(name=col, dtype="numerical", solver="cp", max_n_prebins=20)
    optb.fit(x_tr, target_arr)
    woe_models[col] = optb
    train_fe_out[f"{col}_woe"] = optb.transform(x_tr, metric="woe")
    test_fe_out[f"{col}_woe"]  = optb.transform(test_fe_out[col].values, metric="woe")

print(f"  WoE batch 1: {len(woe_candidates)} features created")

# --- PART 2b: WoE on additional strong/medium variables ---
woe_candidates_2 = [
    'ratings_num_9m_prior', 'ratings_num_12m_prior', 'ratings_num_3m_prior',
    'ratings_num_6m_prior', 'ratings_num_1m_prior', 'rating_pl', 'lang_pl_count',
    'ratings_avg_9m_prior', 'ratings_avg_6m_prior', 'rating_foreign',
    'ratings_avg_1m_prior', 'ratings_avg_3m_prior', 'review_trend',
    'ratings_avg_12m_prior', 'first_review_year_max', 'rating_mean_foreign',
    'tagcat_atmosphere', 'rating_mean_lang_ratio', 'rating_mean_pl',
    'rating_4', 'rating_3', 'tagcat_offerings', 'rating_std',
    'foreign_lang_share', 'review_length_std',
]

for col in woe_candidates_2:
    x_tr = train_fe_out[col].values
    try:
        optb = OptimalBinning(name=col, dtype="numerical", solver="cp", max_n_prebins=20)
        optb.fit(x_tr, target_arr)
        woe_models[col] = optb
        train_fe_out[f"{col}_woe"] = optb.transform(x_tr, metric="woe")
        test_fe_out[f"{col}_woe"]  = optb.transform(test_fe_out[col].values, metric="woe")
    except Exception as e:
        print(f"  WARNING: WoE failed for {col}: {e}")

print(f"  WoE batch 2: {len(woe_candidates_2)} features attempted")

# --- PART 3: Categorical WoE on category_top20 (replaces 21 dummies) ---
cat_cols = [c for c in train_fe_out.columns if c.startswith('cat_')]

def reconstruct_category(row, cat_cols):
    for col in cat_cols:
        if row[col] == 1:
            return col.replace('cat_', '')
    return 'missing'

train_fe_out['_category_top20'] = train_fe_out[cat_cols].apply(
    lambda r: reconstruct_category(r, cat_cols), axis=1)
test_fe_out['_category_top20']  = test_fe_out[cat_cols].apply(
    lambda r: reconstruct_category(r, cat_cols), axis=1)

optb_cat = OptimalBinning(name="category_top20", dtype="categorical", solver="cp")
optb_cat.fit(train_fe_out['_category_top20'].values, target_arr)
train_fe_out['category_woe'] = optb_cat.transform(train_fe_out['_category_top20'].values, metric="woe")
test_fe_out['category_woe']  = optb_cat.transform(test_fe_out['_category_top20'].values, metric="woe")

train_fe_out.drop(columns=cat_cols + ['_category_top20'], inplace=True)
test_fe_out.drop(columns=cat_cols + ['_category_top20'], inplace=True)
print(f"  Category WoE created, {len(cat_cols)} dummies dropped")

# --- PART 4: Interaction features ---
for df_ in [train_fe_out, test_fe_out]:
    df_['vol_x_rating']        = df_['user_ratings_total'] * df_['rating_avg']
    df_['trend_x_vol']         = df_['review_trend'] * df_['user_ratings_total']
    df_['polar_x_amenities']   = df_['polarization'] * df_['tagcat_amenities']
    df_['rating_gap_pl_foreign'] = df_['rating_mean_pl'] - df_['rating_mean_foreign']
    df_['age_x_rating']        = df_['place_age_days'] * df_['rating_avg']
    df_['competition_pressure'] = df_['catch_restaurant_count_500m'] * df_['catch_rating_avg_500m']

print(f"  6 interaction features created")
print(f"After advanced FE — Train: {train_fe_out.shape}, Test: {test_fe_out.shape}")

# =============================================================================
# 5. CORRELATION PRUNING (|ρ| > 0.90)
# =============================================================================
print("\n--- Step 5: Correlation Pruning ---")

features_list = [c for c in train_fe_out.columns if c not in ['restaurant_id', 'status_closed']]

# Compute IV for each feature (to decide which one to keep in correlated pairs)
iv_dict = {}
for col in features_list:
    x = train_fe_out[col].values
    n_unique = pd.Series(x).dropna().nunique()
    dtype = "categorical" if n_unique <= 2 else "numerical"
    try:
        optb_iv = OptimalBinning(name=col, dtype=dtype, solver="cp", max_n_prebins=20)
        optb_iv.fit(x, target_arr)
        bt = optb_iv.binning_table.build()
        iv_dict[col] = bt["IV"].iloc[-1]
    except:
        iv_dict[col] = 0.0

# Protected features: IV >= 0.10 or WoE/interaction features
protected_by_iv = [v for v, iv in iv_dict.items() if iv >= 0.10]
new_feats = [c for c in features_list if c.endswith('_woe') or
             c in ['vol_x_rating', 'trend_x_vol', 'polar_x_amenities',
                   'rating_gap_pl_foreign', 'age_x_rating',
                   'competition_pressure', 'category_woe']]
protected = set(protected_by_iv + new_feats)

# Compute correlation matrix
X_temp = train_fe_out[features_list].fillna(train_fe_out[features_list].median())
corr_matrix = X_temp.corr().abs()

# Identify pairs with |ρ| > 0.90 and decide what to drop
to_drop = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.90:
            v1 = corr_matrix.columns[i]
            v2 = corr_matrix.columns[j]
            v1p, v2p = v1 in protected, v2 in protected
            if v1p and v2p:
                continue
            elif v1p:
                to_drop.append(v2)
            elif v2p:
                to_drop.append(v1)
            else:
                drop = v1 if iv_dict.get(v1, 0) < iv_dict.get(v2, 0) else v2
                to_drop.append(drop)

to_drop = list(set(to_drop))
train_fe_out.drop(columns=to_drop, inplace=True)
test_fe_out.drop(columns=to_drop, inplace=True)

print(f"  Dropped {len(to_drop)} correlated features")
print(f"After pruning — Train: {train_fe_out.shape}, Test: {test_fe_out.shape}")

# =============================================================================
# 6. LASSO FEATURE SELECTION
# =============================================================================
print("\n--- Step 6: LASSO Feature Selection ---")

X_train = train_fe_out.drop(columns=['restaurant_id', 'status_closed'])
X_test  = test_fe_out.drop(columns=['restaurant_id'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Find optimal C for LASSO
C_values = np.logspace(-3, 1, 20)
lasso_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', PowerTransformer(method='yeo-johnson')),
    ('model', LogisticRegression(
        penalty='l1', solver='saga',
        class_weight='balanced',
        max_iter=5000, random_state=42
    ))
])

best_C, best_BA = None, 0
for C in C_values:
    lasso_pipeline.set_params(model__C=C)
    scores = cross_val_score(lasso_pipeline, X_train, target,
                             cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    if scores.mean() > best_BA:
        best_BA = scores.mean()
        best_C  = C

print(f"  Best LASSO C: {best_C:.6f}, BA: {best_BA:.4f}")

# Extract non-zero features
lasso_pipeline.set_params(model__C=best_C)
lasso_pipeline.fit(X_train, target)
coef = lasso_pipeline.named_steps['model'].coef_[0]
selected_features = X_train.columns[coef != 0].tolist()

print(f"  Features selected: {len(selected_features)} / {X_train.shape[1]}")

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

# =============================================================================
# 7. FINAL MODEL — PowerTransformer + ElasticNet
# =============================================================================
print("\n--- Step 7: Training Final Model ---")

final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  PowerTransformer(method='yeo-johnson')),
    ('model',   LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=0.1,
        l1_ratio=0.2,
        class_weight='balanced',
        max_iter=5000,
        random_state=42
    ))
])

# Cross-validated balanced accuracy (expected performance)
cv_scores = cross_val_score(final_pipeline, X_train_sel, target,
                            cv=cv, scoring='balanced_accuracy', n_jobs=-1)
print(f"  CV Balanced Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Fit on full training data
final_pipeline.fit(X_train_sel, target)
print("  Model trained on full dataset")

# =============================================================================
# 8. PREDICTIONS AND SUBMISSION
# =============================================================================
print("\n--- Step 8: Generating Predictions ---")

THRESHOLD = 0.470

test_proba = final_pipeline.predict_proba(X_test_sel)[:, 1]
test_preds = (test_proba >= THRESHOLD).astype(int)

print(f"  Threshold: {THRESHOLD}")
print(f"  Class 0 (Open):   {(test_preds == 0).sum()}")
print(f"  Class 1 (Closed): {(test_preds == 1).sum()}")
print(f"  % Closed: {test_preds.mean()*100:.1f}%")

submission = pd.DataFrame({
    'restaurant_id': test_id,
    'status_closed': test_preds
})

submission.to_csv('submission.csv', index=False)
print("\n✅ submission.csv saved successfully")
print(f"   Shape: {submission.shape}")
print(submission.head())
