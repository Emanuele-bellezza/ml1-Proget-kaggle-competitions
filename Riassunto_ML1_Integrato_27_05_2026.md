# Riassunto Integrato ML1 — Classificazione + Regressione
## Data: 27 maggio 2026

---

## STATO ATTUALE

| | Classificazione | Regressione |
|---|---|---|
| **Task** | Predire chiusura ristorante (status_closed) | Predire stipendio developer (annual.pay.usd) |
| **Metrica** | Balanced Accuracy (BA) | RMSE in USD |
| **Posizione Kaggle** | 9° posto | 11°–12° posto |
| **Score pubblico** | BA = 0.686 | RMSE = $34,090 |
| **Miglior CV interno** | BA = 0.6852 | RMSE = $33,068 |

---

# PARTE 1 — CLASSIFICAZIONE: Predizione Chiusura Ristoranti

## Dataset
- Train: 33.296 osservazioni × 86 colonne (incluso target)
- Test: 8.325 osservazioni × 85 colonne
- Target: `status_closed` — fortemente sbilanciato: **90.2% aperti / 9.8% chiusi**
- Soluzione adottata: `class_weight='balanced'` in tutti i modelli
- Algoritmi ammessi: Logistic Regression, KNN, Ridge, LASSO, ElasticNet, SVC (vincolo del corso)

---

## STEP 1 — EDA (Exploratory Data Analysis)

- 30 coppie di variabili con correlazione |ρ| > 0.90
- 15+ variabili con skewness > 3 (distribuzioni fortemente asimmetriche)
- `price_level`: 50.9% NaN | `tagcat_payment_options`: 71.6% NaN | `category_top20`: 765 NaN
- `weekends_only` / `workdays_only`: scoperte come stringhe ("True"/"False") invece che boolean — fix critico applicato
- Target sbilanciato confermato: richiede metriche e modelli adeguati

---

## STEP 2 — Data Preparation

Strategia missing values divisa in 6 gruppi logici:
- Variabili di rating con 0 recensioni → imputazione con 0
- Variabili temporali → mediana
- Variabili linguistiche → mediana
- Variabili di orario → mediana
- Variabili categoriche → categoria "missing" separata
- `tagcat_payment_options` (71.6% NaN) → mediana

Fix critico `weekends_only` / `workdays_only`:
```python
df['weekends_only'] = df['weekends_only'].astype(str).str.strip()
    .replace({'True': 1, 'False': 0, 'nan': np.nan}).astype(float)
```

Output: `restaurants_train_clean.csv` / `restaurants_test_clean.csv` — **109 feature**

---

## STEP 3 — Feature Engineering Base

8 nuove feature create:
- **Anelli marginali POI** (4): densità in anelli concentrici (100–200m, 200–500m, 500–1000m, 1000–2000m)
- **Anelli marginali catch** (2): ristoranti competitori in anelli differenziali
- `place_age_days_sq`: termine quadratico per catturare effetto non-lineare dell'età
- `bayesian_rating`: rating bayesiano (attenua valutazioni con poche recensioni)
- `review_trend`: tendenza recente delle recensioni
- `rating_cv`: coefficiente di variazione del rating
- `polarization`: indice di polarizzazione delle valutazioni
- `weekend_dependency`: dipendenza dal traffico del weekend

Output: `restaurants_train_fe.csv` / `restaurants_test_fe.csv` — **121 feature**

---

## STEP 3b — Feature Engineering Avanzato (WoE + Interazioni)

### Parte 1 — Information Value (IV)
Calcolato l'IV di tutte le 121 feature con `OptimalBinning`:

| Categoria | N. variabili |
|---|---|
| Strong (IV > 0.3) | 14 |
| Medium (IV 0.1–0.3) | 19 |
| Weak (IV 0.02–0.1) | 36 |
| Not useful (IV < 0.02) | 50 |

Scoperta importante: quasi tutte le dummy `cat_*` hanno IV = 0 → nessun potere predittivo grezzo.

### Parte 2 — WoE (Weight of Evidence) su variabili Strong e Medium
Il WoE trasforma variabili continue in valori che esprimono direttamente la relazione con il target.
È particolarmente utile per la Logistic Regression perché linearizza relazioni non-lineari.

**Procedura:**
1. Fit di `OptimalBinning` su X_train (mai su validation/test)
2. Trasformazione applicata sia a train che a test
3. Variabile WoE creata **accanto** all'originale (non in sostituzione)
4. Il LASSO a valle decide quale delle due tenere

**Variabili con WoE:**
- 6 variabili originali (batch 1): `user_ratings_total`, `place_age_days`, `tagcat_amenities`, `review_length_avg`, `polarization`, `review_has_text_pct`
- 12 variabili Strong aggiuntive (batch 2): tutte le `ratings_num_*m_prior`, `rating_pl`, `lang_pl_count`, `ratings_avg_*m_prior`, `rating_foreign`
- 13 variabili Medium aggiuntive (batch 2): `review_trend`, `ratings_avg_12m_prior`, `first_review_year_max`, `rating_mean_foreign`, `tagcat_atmosphere`, `rating_mean_lang_ratio`, `rating_mean_pl`, `rating_4`, `rating_3`, `tagcat_offerings`, `rating_std`, `foreign_lang_share`, `review_length_std`

### Parte 3 — WoE categorico su category_top20
Le 21 dummy di `category_top20` avevano IV = 0 individualmente.
Applicato WoE categorico: ricostruzione della variabile originale → OptimalBinning categorico → 5 gruppi con event rate da 4.9% a 19.1% → creata `category_woe` (IV = 0.09).
Le 21 dummy originali sono state droppate.

### Parte 4 — Interazioni
6 interazioni tra variabili ad alto IV:
`vol_x_rating`, `trend_x_vol`, `polar_x_amenities`, `rating_gap_pl_foreign`, `age_x_rating`, `competition_pressure`

Output: `restaurants_train_fe2.csv` / `restaurants_test_fe2.csv` — **138 feature**

---

## STEP 3c — Revisione Strategia di Imputazione

Confronto di 3 pipeline in cross-validation (stesso modello ElasticNet):

| Strategia | BA (CV) |
|---|---|
| **A — Mediana (baseline)** | **0.6709** ✓ |
| B — KNN Imputer | 0.6690 |
| C — Mediana + Missing Flags | 0.6707 |

**Decisione: mediana confermata come strategia ottimale.**
La complessità aggiuntiva (KNN imputer, flag aggiuntivi) non porta benefici.

---

## STEP 4 — Feature Selection

### Correlation Pruning
Rimozione delle ridondanze tra variabili con |ρ| > 0.90.

**Regole di protezione obbligatorie:**
- MAI droppare variabili con IV ≥ 0.10 (Strong o Medium)
- MAI droppare feature WoE
- MAI droppare le interazioni
- Solo variabili deboli e ridondanti vengono eliminate

Senza queste protezioni il pruning eliminava variabili fondamentali come `user_ratings_total` e `place_age_days`, causando un peggioramento della BA.

### Feature Selection con LASSO
Il LASSO (LogisticRegression con penalty='l1') è usato **esclusivamente come selettore**, non come modello finale.

**Procedura:**
1. Ricerca del C ottimale via cross-validation (C = 0.206914)
2. Fit del LASSO → identificazione dei coefficienti azzerati
3. Rimozione delle feature azzerate dal dataset
4. Dataset finale passato al modello ottimale (ElasticNet o Ridge)

**Risultato:** 106 feature selezionate

**Conferma della logica WoE:** il LASSO ha azzerato le originali e tenuto i WoE in tutti i casi in cui entrambe erano presenti → conferma che il WoE cattura meglio il segnale per la Logistic Regression.

---

## STEP 5–6 — Confronto Modelli

### Confronto Scaler × Penalty (pipeline con 106 feature)

| Combinazione | BA (CV) | Parametri |
|---|---|---|
| **PowerTransformer + L2** | **0.6852** ★ | C=0.1 |
| PowerTransformer + ElasticNet | 0.6849 | C=0.1, l1=0.5 |
| PowerTransformer + L1 | 0.6838 | C=0.1 |
| StandardScaler + ElasticNet | 0.6791 | C=0.1, l1=0.75 |
| StandardScaler + L2 | 0.6787 | C=10.0 |
| StandardScaler + L1 | 0.6784 | C=10.0 |

**Osservazione chiave: PowerTransformer domina StandardScaler di ~+6 pp.**
Yeo-Johnson normalizza le distribuzioni asimmetriche prima della regressione, migliorando significativamente la stima dei coefficienti.

### Confronto completo modelli

| Modello | BA (CV) | Note |
|---|---|---|
| PowerTransformer + L2 ★ | 0.6852 | Vincitore attuale |
| PowerTransformer + ElasticNet | 0.6849 | Equivalente |
| Logistic pura (no penalty) | 0.6786 | |
| SVC RBF (C=0.1) | 0.6760 | Con PowerTransformer |
| KNN (k=5) | 0.5175 | Non adatto |

### Cosa non ha funzionato
- **Imputazione**: KNN imputer e missing flags peggiorano o equivalgono alla mediana → mediana confermata
- **StandardScaler**: sistematicamente inferiore a PowerTransformer in tutte le configurazioni → scartato

---

## PIPELINE FINALE

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', PowerTransformer(method='yeo-johnson')),
    ('model', LogisticRegression(
        penalty='l2', solver='lbfgs', C=0.1,
        class_weight='balanced', max_iter=5000, random_state=42
    ))
])
```

- **Feature:** 106 (post LASSO selection)
- **Threshold:** 0.535
- **Kaggle pubblico:** BA = 0.682 (in attesa submission con threshold ottimale)
- **CV interno:** BA = 0.6852

---

## SLIDE CONFRONTO 3 PIPELINE — CLASSIFICAZIONE

| Pipeline | Feature | BA (CV) | Note |
|---|---|---|---|
| FE base senza WoE | 121 | 0.6695 | Baseline pre-FE avanzato |
| WoE batch 1 (6 variabili) + LASSO | 84 | 0.6774 | Primo salto significativo |
| WoE completo (Strong+Medium) + LASSO | 106 | 0.6852 ★ | Pipeline finale |

---

---

# PARTE 2 — REGRESSIONE: Predizione Stipendio Developer

## Dataset
- Train: 2.512 osservazioni × 41 colonne | Test: 628 osservazioni
- Target: `annual.pay.usd` — distribuzione fortemente asimmetrica (skewness ~33)
- Algoritmi: Linear Regression, Ridge, Lasso, ElasticNet, KNN, SVR (RBF + Polynomial)

---

## STEP 1 — EDA

- Media: $49.713 | Mediana: $40.828 | Max: $4.773.360
- 503 osservazioni sotto $10k → molti stipendi mensili non annualizzati
- Solo 5 osservazioni sopra $500k

**Top variabili correlate con log(stipendio):**
- `coding.years.total` → |corr| = 0.286
- `coding.years.professional` → |corr| = 0.278
- `experience.years` → |corr| = 0.273
- `age.group` → |corr| = 0.218

**Top tecnologie per correlazione:** Terraform (+0.123), Bash/Shell (+0.103), Azure (+0.074)

---

## STEP 2 — Data Cleaning

- **Fix stipendi mensili:** 378 osservazioni con pay < $10k ma ×12 nel range [10k, 500k] → moltiplicati ×12
- **Rimozione outlier:** 130 osservazioni fuori da [10k, 500k] USD → 2.382 osservazioni finali
- **Fix inconsistenze logiche:** `coding.years.professional > coding.years.total` → NaN | under-24 con >20 anni di codice → NaN. 104 inconsistenze corrette
- **Trasformazione target:** `log1p(annual.pay.usd)` — skewness da ~33 a ~1

---

## STEP 3 — Split Stratificato

- Split 80/20 eseguito **prima** di qualsiasi preprocessing → zero data leakage
- Stratificazione su quintili del target logaritmico
- Train: 1.905 obs | Validation: 477 obs

---

## STEP 4 — Feature Engineering (351 feature totali)

### Encoding
- **Ordinali → numerico:** `age.group`, `education`, `company.size`, `ai.sentiment`, ecc.
- **Nominali → One-Hot:** `region`, `dev.role`, `employment.type`, `industry`, `work.location` (fit solo su X_train)
- **Multi-select → binary indicators:** `prog.languages`, `databases`, `cloud.platforms`, `dev.tools`, `web.frameworks` → top-15 item per correlazione + colonna count
- **Missing flags:** flag binario `__missing` per colonne con >30% NaN

### Feature costruite ad-hoc

| Feature | Descrizione |
|---|---|
| `region_target_enc` | Mediana log-stipendio per regione (fit solo su X_train) |
| `dev.role_target_enc` | Mediana log-stipendio per ruolo developer |
| `industry_target_enc` | Mediana log-stipendio per industry |
| `exp_bucket` | Esperienza professionale in 5 livelli discreti (0–4) |
| `exp_professional_sq` | Termine quadratico esperienza (cattura curva carriera) |
| `age_squared` | Termine quadratico età (picco 35-44, poi cala) |
| `manager_x_company` | is_manager × company_size_num |
| `total_senior` | Count tecnologie senior (Terraform, K8s, AWS, Redis…) |
| `senior_x_top_region` | is_senior × is_top_region |
| `remote_senior_top` | is_remote × is_senior × is_top_region |
| `is_young_senior` | age 25-34 AND exp ≥ 7 anni |

**Totale: 351 feature** (308 binarie + 43 continue)

---

## STEP 5 — Feature Selection (3 approcci confrontati)

### Approccio 1 — PRIMA (351 feature, nessuna selezione)
Tutte le feature generate, nessun filtro.

### Approccio 2 — Selezione STATISTICA (150 feature) ★ VINCITORE
Criterio doppio per tipo di variabile:
- **Variabili continue** → Pearson |corr| < 0.01 → rimosse (9 rimosse)
- **Variabili binarie** → ANOVA F-statistic con p ≥ 0.05 → rimosse (189 rimosse)
- **Ridondanti manuali** → rimosse 6: `coding.years.total`, `age.group`, `is_top_region`, `total_junior`, `senior_ratio`, `coding.years.professional`

**Perché ANOVA per le binarie?**
Una variabile binaria con P(1)=5% ha correlazione di Pearson massima ≈ 0.22 per limite matematico — non è inutile, è rara. L'ANOVA F-statistic misura se la media del target è diversa tra i due gruppi (0 vs 1) e non soffre di questo limite. Esempio: `senior_x_top_region` ha Pearson=0.006 (sembrerebbe rumore) ma F-stat=152 (segnale fortissimo).

Risultato: **da 351 a 150 feature (-57%)**

### Approccio 3 — Selezione LASSO (300 feature, alpha=0.001)
LASSO usato come selettore, non come modello predittivo.
alpha=0.001 permissivo → azzera solo **51 feature su 351**.
Top feature per coefficiente: `exp_bucket` (0.163), `coding.years.professional` (0.131), `exp_professional_sq` (0.110), `region_target_enc` (0.091).

---

## STEP 6 — Confronto Modelli (3 blocchi × 7 modelli)

Pipeline standard per tutti i modelli:
```
VarianceThreshold(0.01) → StandardScaler → Modello
```

Pipeline speciale solo per SVR:
```
VarianceThreshold(0.01) → PowerTransformer(Yeo-Johnson) → StandardScaler → SVR
```

### SLIDE CONFRONTO 3 PIPELINE — REGRESSIONE (Validation RMSE in USD)

| Modello | PRIMA (351 feat) | STAT (150 feat) | LASSO (300 feat) | Migliore |
|---|---|---|---|---|
| Linear Regression | $35,742 | $33,777 | $35,267 | STAT |
| Ridge | $33,766 | $33,682 | $33,829 | STAT |
| Lasso | $34,607 | $33,773 | $34,454 | STAT |
| ElasticNet | $34,514 | $33,803 | $34,266 | STAT |
| KNN | $37,218 | $37,313 | $36,679 | LASSO |
| **SVR RBF** | $33,425 | **$33,068 ★** | $33,806 | **STAT** |
| SVR Polynomial | $36,041 | $45,431 | $35,313 | PRIMA |

**Miglior modello assoluto: SVR RBF con selezione statistica (150 feature)**
Parametri ottimali: C=5, gamma=0.0001, epsilon=0.1

**Osservazioni chiave:**
- La selezione statistica migliora 5 modelli su 7
- Da 351 a 150 feature (-57%) con beneficio netto
- LASSO permissivo (alpha=0.001) azzera solo 51 feature → quasi nessun beneficio
- SVR Polynomial con selezione statistica crolla a $45,431 → da evitare

---

## STEP 7 — Validazione Robusta

- 5-Fold CV ripetuta (3 ripetizioni → 15 fold totali)
- RMSE log medio: 0.5452 | std: 0.0192
- Range: [0.5124 — 0.5841]

---

## STEP 8 — Retrain e Submission

- Retrain su tutto il training set pulito (2.382 osservazioni)
- Dataset finale: 2.382 × 172 feature (train) | 628 × 172 feature (test)
- Predizioni test: Min $13.010 | Mediana $45.414 | Max $136.822
- Kaggle pubblico RMSE: ~$34,090

### Diagnosi degli errori per fascia

| Fascia stipendio | N obs | MAE | Errore % | Giudizio |
|---|---|---|---|---|
| < $20k | 147 | $26,924 | 344% | CRITICO |
| $20k–$40k | 105 | $10,813 | 36.2% | Alto |
| $40k–$70k | 133 | $15,392 | 28.1% | OK |
| $70k–$120k | 97 | $35,874 | 40.0% | Alto |
| > $120k | 20 | $84,376 | 58.0% | CRITICO |

**Causa strutturale:** le fasce estreme hanno pochissimi esempi in training — il modello non ha mai visto abbastanza stipendi < $20k o > $120k per impararli bene.

---

## PIPELINE FINALE REGRESSIONE

```python
Pipeline([
    ('variance', VarianceThreshold(0.01)),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler()),
    ('model', SVR(kernel='rbf', C=5, gamma=0.0001, epsilon=0.1))
])
```

- **Feature:** 150 (post selezione statistica)
- **Kaggle pubblico:** RMSE = $34,090
- **CV interno:** RMSE = $33,068

---

## LEZIONI APPRESE (entrambi i progetti)

**Classificazione:**
- PowerTransformer è il singolo fattore più impattante (+6 pp vs StandardScaler)
- WoE trasforma variabili continue in segnali più puliti per la Logistic Regression
- Le dummy di category_top20 avevano IV=0 individualmente — il WoE categorico ha estratto segnale (IV=0.09)
- Il LASSO ha confermato che i WoE sono superiori alle originali (ha azzerato le originali dove serviva)
- Il pruning per correlazione deve proteggere le variabili importanti (IV alto)
- L'imputazione con mediana è sufficiente — KNN e missing flags non migliorano

**Regressione:**
- La trasformazione del target (log1p) è fondamentale per gestire la skewness
- ANOVA F-statistic è superiore a Pearson per le variabili binarie rare
- SVR con PowerTransformer beneficia della normalizzazione (kernel RBF usa distanze euclidee)
- SVR Polynomial è instabile con selezione aggressiva delle feature
- Le fasce estreme del target sono strutturalmente difficili da predire

---

*Riassunto generato il 27/05/2026 — da usare come context per la generazione della presentazione*
