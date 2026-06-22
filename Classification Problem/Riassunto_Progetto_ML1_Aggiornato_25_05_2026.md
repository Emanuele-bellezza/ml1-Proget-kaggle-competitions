Riassunto Progetto ML1 — Restaurant Survival Classification (AGGIORNATO 25/05/2026)
==================================================================================

Stato attuale: 9° in classifica Kaggle (BA = 0.686)
Stato precedente: 26° in classifica Kaggle (BA = 0.663)

===================================================================================
CONTESTO
===================================================================================
Progetto universitario ML1 (Prof. Piotr Wójcik, a.y. 2025/2026).
Obiettivo: prevedere status_closed (chiusura permanente ristorante) su dataset di locali polacchi.
Metrica: balanced accuracy.
Vincolo: solo modelli trattati a lezione (Logistic Regression, KNN, Ridge, LASSO, Elastic Net, SVC).

Dataset:
- Train: 33.296 osservazioni, 86 colonne (incluso target)
- Test: 8.325 osservazioni, 85 colonne
- Target: status_closed — fortemente sbilanciato: 90.2% classe 0 / 9.8% classe 1
- class_weight='balanced' in tutti i modelli

Primo in classifica: 0.695 → gap da colmare: ~0.9 pp

===================================================================================
PIPELINE DI LAVORO COMPLETA (aggiornata)
===================================================================================

1. EDA (01_EDA_restaurants.py) — INVARIATO
   - 30 coppie con ρ > 0.90
   - 15+ variabili con skewness > 3
   - weekends_only/workdays_only erano stringhe, non boolean
   - price_level 50.9% NaN, category_top20 765 NaN, tagcat_payment_options 71.6% NaN

2. Data Preparation (02_data_preparation.py) — CORRETTO
   - Strategia missing values in 6 gruppi (invariata)
   - FIX CRITICO: weekends_only/workdays_only richiedono:
     df['weekends_only'] = df['weekends_only'].astype(str).str.strip().replace({'True': 1, 'False': 0, 'nan': np.nan}).astype(float)
   - FIX CRITICO: assicurarsi che to_csv salvi nella cartella Data/ (non nella root)
   - Output: restaurants_train_clean.csv, restaurants_test_clean.csv (109 feature)

3. Feature Engineering Base (03_feature_engineering.py) — INVARIATO
   8 feature create:
   - Anelli marginali POI (4): poi_ring_100_200, poi_ring_200_500, poi_ring_500_1000, poi_ring_1000_2000
   - Anelli marginali catch (2): catch_ring_500_1000, catch_ring_1000_2000
   - place_age_days_sq, bayesian_rating, review_trend, rating_cv, polarization, weekend_dependency
   - Output: restaurants_train_fe.csv, restaurants_test_fe.csv (121 feature)

3b. ADVANCED FEATURE ENGINEERING (03b — NUOVO)
    Input: restaurants_train_fe.csv, restaurants_test_fe.csv

    PARTE 1 — Information Value (IV):
    - Calcolato IV di tutte le 121 feature con OptimalBinning
    - Risultato: 14 Strong (IV>0.3), 19 Medium (0.1-0.3), 36 Weak (0.02-0.1), 50 Not useful (<0.02)
    - Top 5: ratings_num_9m_prior (0.439), ratings_num_12m_prior (0.431), ratings_num_3m_prior (0.423),
      ratings_num_6m_prior (0.420), user_ratings_total (0.391)
    - Scoperta chiave: quasi tutte le dummy cat_* hanno IV = 0, is_new_restaurant IV = 0
    - IV salvati in Data/03b_iv_results.csv

    PARTE 2 — WoE su 6 variabili numeriche top:
    - user_ratings_total_woe (IV=0.39, 12 bin, relazione monotona fortissima)
    - place_age_days_woe (IV=0.25, 10 bin, missing hanno event rate 29%!)
    - tagcat_amenities_woe (IV=0.27, 8 bin, relazione monotona)
    - review_length_avg_woe (IV=0.09, 11 bin)
    - polarization_woe (IV=0.30, 8 bin, curva a U invertita)
    - review_has_text_pct_woe (IV=0.10, 9 bin)
    - Strategia: WoE creati ACCANTO alle originali, LASSO decide quale tenere

    PARTE 3 — WoE su category_top20 (categorica):
    - Ricostruita category_top20 dalle 21 dummy
    - WoE categorico ha aggregato in 5 gruppi sensati:
      * Rischio basso: Hot dogs, Family restaurant, Vietnamese, Polish, Sushi (4.9% chiusura)
      * Rischio alto: Fast food + missing (19.1% chiusura)
      * Rischio alto: Turkish/Burgers/Kebab (13.9% chiusura)
    - IV totale = 0.092 (Weak, ma molto meglio delle 21 dummy a IV=0)
    - Droppate tutte le 21 dummy, creata 'category_woe'

    PARTE 4 — 6 interazioni:
    - vol_x_rating = user_ratings_total × rating_avg
    - trend_x_vol = review_trend × user_ratings_total
    - polar_x_amenities = polarization × tagcat_amenities
    - rating_gap_pl_foreign = rating_mean_pl - rating_mean_foreign
    - age_x_rating = place_age_days × rating_avg
    - competition_pressure = catch_restaurant_count_500m × catch_rating_avg_500m

    Output: restaurants_train_fe2.csv, restaurants_test_fe2.csv (115 colonne train, 114 test)

3c. REVISIONE IMPUTAZIONE (03c — NUOVO)
    Confronto 3 strategie in CV (stesso ElasticNet):
    - Pipeline A — Mediana (baseline): BA = 0.6709
    - Pipeline B — KNN Imputer: BA = 0.6690
    - Pipeline C — Mediana + Missing Flags: BA = 0.6707
    → DECISIONE: mediana confermata migliore, nessuna modifica

4. CORRELATION PRUNING CORRETTO (04 — AGGIORNATO)
   Input: restaurants_train_fe2.csv

   REGOLE DI PROTEZIONE:
   - MAI droppare variabili con IV >= 0.10 (Strong/Medium)
   - MAI droppare feature WoE (convivono con originali)
   - MAI droppare interazioni
   - Solo variabili deboli ridondanti vengono droppate

   ERRORE PRECEDENTE (v1): aveva assegnato IV=0.5 di default alle feature nuove,
   causando il drop di user_ratings_total e place_age_days → BA peggiorata a 0.6670.
   CORRETTO nella v2 con le regole di protezione → BA risalita a 0.6714+.

   44 coppie con |ρ| > 0.90 trovate.
   Molte skippate perché entrambe protette.

4b. FEATURE SELECTION LASSO (04b — NUOVO)
    - LASSO usato SOLO come strumento di selezione (non come modello finale)
    - C ottimale LASSO: 0.078476, BA = 0.6760
    - Feature tenute: 84 (coeff ≠ 0)
    - Feature azzerate: 13

    Feature azzerate da LASSO (conferma della logica):
    - user_ratings_total, place_age_days, tagcat_amenities, rating_pl → sostituite dai rispettivi WoE
    - vol_x_rating, age_x_rating, competition_pressure → interazioni non informative
    - is_new_restaurant → confermato inutile (IV=0)
    - review_trend, types_count, type_bar, price_level_2 → deboli
    - rating_mean_lang_ratio → debole

    Confronto: Tutte (97 feat) BA=0.6714 vs LASSO (84 feat) BA=0.6714 → equivalenti, meno feature = meglio

    Dataset finale: restaurants_train_final.csv, restaurants_test_final.csv (84 feature)

5-6. MODELLAZIONE (05-06 — AGGIORNATO)
     Ricerca sistematica: 2 Scaler × 3 Penalty

     CLASSIFICA FINALE:
     1. PowerTransformer + ElasticNet  → BA = 0.6774 (C=0.1, l1_ratio=0.5)  ★ VINCITORE
     2. PowerTransformer + L2          → BA = 0.6771 (C=0.05)
     3. PowerTransformer + L1          → BA = 0.6769 (C=10.0)
     4. StandardScaler + L2            → BA = 0.6631 (C=10.0)
     5. StandardScaler + ElasticNet    → BA = 0.6630 (C=10.0, l1=0.5)
     6. StandardScaler + L1            → BA = 0.6630 (C=10.0)

     Nota: PowerTransformer domina (+1.4 pp vs StandardScaler)
     Nota: i 3 modelli con PowerTransformer sono quasi equivalenti
     Nota: SVC RBF (StandardScaler) aveva BA=0.6649 — non testato con PowerTransformer

7. Model Evaluation (07 — AGGIORNATO)
   - Threshold analysis da rifare con i nuovi parametri (C=0.1, l1_ratio=0.5)

8. Submission Kaggle (08 — AGGIORNATO)
   - Kaggle score: 0.686 (9° posto)
   - Gap dal primo: ~0.9 pp (primo = 0.695)

===================================================================================
ARCHITETTURA PIPELINE FINALE
===================================================================================
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', PowerTransformer(method='yeo-johnson')),
    ('model', LogisticRegression(
        penalty='elasticnet', solver='saga', C=0.1,
        l1_ratio=0.5, class_weight='balanced', max_iter=5000, random_state=42
    ))
])

===================================================================================
REGOLE NON NEGOZIABILI
===================================================================================
- Zero data leakage: scaling, imputazione, trasformazioni dentro Pipeline
- Validazione: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) con scoring='balanced_accuracy'
- Solo modelli trattati a lezione
- class_weight='balanced' su tutti i modelli (tranne KNN)

===================================================================================
FILE PRODOTTI
===================================================================================
Data/:
  - restaurants_train_clean.csv / restaurants_test_clean.csv (109 feature, post step 2)
  - restaurants_train_fe.csv / restaurants_test_fe.csv (121 feature, post step 3)
  - restaurants_train_fe2.csv / restaurants_test_fe2.csv (115/114 feature, post step 3b)
  - restaurants_train_final.csv / restaurants_test_final.csv (84 feature, post LASSO selection)
  - 03b_iv_results.csv (IV di tutte le feature)
  - submission.csv
  - Vari grafici PNG e CSV risultati

===================================================================================
PROSSIME DIREZIONI PER MIGLIORARE (gap: ~0.9 pp dal primo)
===================================================================================
1. Griglia più fine su C e l1_ratio — testare C tra 0.05 e 0.2, l1_ratio tra 0.3 e 0.7
2. Threshold tuning — verificare se il threshold ottimale del CV migliora anche su Kaggle
3. WoE su altre variabili — rating_std, foreign_lang_share, ratings_avg_1m_prior hanno IV Medium
   con molti bin → potrebbero beneficiare del WoE
4. Rimuovere il pruning — dare tutte le 113 feature post-FE2 al modello senza pruning/LASSO,
   lasciando che ElasticNet selezioni da solo
5. SVC con PowerTransformer — non ancora testato, potrebbe sorprendere
6. Combinare predizioni — media delle probabilità di più modelli (ensemble semplice)

===================================================================================
LEZIONI APPRESE
===================================================================================
- PowerTransformer è il singolo fattore più impattante (+1.4 pp vs StandardScaler)
- WoE trasforma variabili continue in segnali più puliti per la logistic regression
- Le dummy di category_top20 avevano IV=0, il WoE categorico ha estratto segnale (IV=0.09)
- Il LASSO ha confermato che i WoE sono superiori alle originali (ha azzerato le originali)
- Il pruning per correlazione deve proteggere le variabili importanti (IV alto)
- L'imputazione con mediana è sufficiente — KNN e missing flags non migliorano
- weekends_only/workdays_only: attenzione al formato stringa con .astype(str).str.strip().replace()
- Attenzione ai percorsi file: to_csv('Data/...') non to_csv('...')

===================================================================================
STILE DI LAVORO CONCORDATO CON L'UTENTE
===================================================================================
- Le risposte devono spiegare i passaggi, non prendere decisioni in autonomia
- Flow collaborativo: proporre → discutere → decidere insieme → generare codice
- NON generare file senza che venga chiesto
- Se chiesto, generare SOLO il codice senza dire altro
- L'utente runna il codice localmente e incolla i risultati per l'analisi
- Il modello deve essere facile da spiegare in presentazione
