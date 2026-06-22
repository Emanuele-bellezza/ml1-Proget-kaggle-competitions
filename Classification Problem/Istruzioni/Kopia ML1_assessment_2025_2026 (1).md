## Machine Learning 1 

## Assessment rules 

## Piotr Wójcik, Szymon Lis, Michał Woźniak 

academic year 2025/2026 

## Table of Contents 

General information ......................................................................................................... 1 Datasets .......................................................................................................................... 1 `restaurants` for **classification** .................................................................................. 1 Files Provided ........................................................................................................... 2 `annual.pay.usd` for **regression** ................................................................................. 2 Files Provided ........................................................................................................... 2 Requirements ............................................................................................................... 2 Competitions on kaggle ................................................................................................... 3 Various algorithms ........................................................................................................... 3 Selection of the best algorithm ........................................................................................ 3 Performance measure ..................................................................................................... 4 Points............................................................................................................................... 4 Presentations ................................................................................................................... 4 Important dates again ...................................................................................................... 5 

## General information 

In teams of **at most 2 persons students** will work on two practical machine learning projects - one for **regression** and one for **classification** . Please inform the lecturer about the team members by email pwojcik@wne.uw.edu.pl or sm.lis@student.uw.edu.pl **the latest by midnight 2026-04-20** . 

## Datasets 

The data is **exactly the same for all teams** . 

## `restaurants` for **classification** 

Your task is to apply various ML algorithms and tools **discussed on Machine Learning 1 course** (see the rules below) to build a model **explaining the restaurant survival** 

based on the **training sample** and generate predictions for **all observations** from the **test sample** . 

More details are provided here: https://www.kaggle.com/competitions/ml-1-2026-task-1predicting-restaurant-survival-classification 

## Files Provided 

- `restaurants_train.csv` - training data contains 33296 observations and 86 columns along with the target variable **status_closed** . 

- `restaurants_test.csv` - test data contains 8325 observations and 85 columns **without** the target variable. 

- `restaurant_sample_submission.csv` - sample submission in the corerct format file containing 8325 observations and 2 columns: `restaurant_id` and `status_closed` (prediction). 

`annual.pay.usd` for **regression** 

Your task is to apply various ML algorithms and tools discussed on Machine Learning 1 course (see the detailed requirements below) to build a model **predicting the annual compensation of software developers** based on the **training sample** and generate predictions for **all observations** from the **test sample** . 

More details are provided here: https://www.kaggle.com/competitions/ml-1-2026-task-2developer-salary-prediction-regression 

## Files Provided 

- `train.csv` – training data contains 2,512 observations and 41 columns along with the target variable **annual.pay.usd** . 

- `test.csv` – test data contains 628 observations and 41 columns (40 features + `id` ) **without** the target variable. 

- `sample_submission.csv` – sample submission file in the correct format. 

## Requirements 

## 1. **Exploratory Data Analysis:** 

- Analyze the dataset to identify key patterns, correlations, and potential challenges introduced by the additional variables. 

- Visualize distributions and relationships among variables. 

## 2. **Feature Engineering:** 

- Consider transformation or scaling of variables as needed. 

- Evaluate the impact of the supplementary variables on model performance. 

## 3. **Modeling:** 

- Build, train, and compare multiple regression models. 

- Optimize model hyperparameters using cross-validation. 

## 4. **Predictions:** 

- Generate predictions for all observations in the test dataset. 

- Document model performance and reasoning behind the selected approach. 

## 5. **Documentation:** 

- Provide a clear explanation of your analysis, modeling choices, and any challenges faced while integrating the additional variables. 

## Competitions on kaggle 

Both datasets and their descriptions are available on kaggle: 

- `restaurants` for classification: https://www.kaggle.com/competitions/ml-1-2026task-1-predicting-restaurant-survival-classification 

- `annual.pay.usd` for regression: https://www.kaggle.com/competitions/ml-1-2026task-2-developer-salary-prediction-regression 

In both cases you are required to upload the file with final predictions that **only includes** the observation **id** and the **predicted value of the outcome variable** (check details on kaggle). 

**IMPORTANT!** Groups that do NOT upload their submissions on time **will not be graded** . 

- in case of each dataset **30% of the test data** will be used for the **Public Leaderboard** during the competition, 

- the remaining **70% of the test data** will be used for the **final Private Leaderboard** - used in the **course assessment** 

## Various algorithms 

For each of the datasets please consider and compare **at least 3 different ML algorithms discussed in the ML1 course** (e.g. linear/logistic regression, KNN, LASSO, ridge, elastic net, SVM/SVR with various kernel functions). Do **NOT** use any other algorithms beyond the scope of the ML1 course. Still you can apply other known tools - data rebalancing, feature engineering, feature selection. 

## Selection of the best algorithm 

The choice of the final algorithm applied to generate predictions should be **clearly explained** in the presentation. 

HINT !!!!! Use the internal division of the training data into train/validation/test samples to make sure that you correcly assess the performance of your models on the new data. 

## Performance measure 

The performance of predictions will be based on: 

1. `balanced accuracy` for the **restaurants** dataset 

2. `RMSE` for the **annual.pay.usd** dataset 

Please report the **expected** value of a particular performance measure (expectation for the test sample) in your presentation. It can be based on the public leaderboard from kaggle. 

## Points 

In total **60 points** can be collected - 30 for each project: 

- **presentation in class** - its structure, way of presenting, etc. ( **10 pts** for each project) 

- **presentation contents** - assessed by the lecturer after you present in class ( **10 pts** for each project) 

- **out-of-sample** performance ( **10 pts** for each project): 

   - `10` if predictive performance in top quartile group (best), 

   - `7.5` if predictive performance in the 2nd quartile group (good), 

   - `5` if predictive performance in the 3rd quartile group (below average), 

   - `2.5` if predictive performance in the 4th quartile group (unlucky), 

## Presentations 

The competitions on kaggle are open **until 23:55 2026-05-28** . By this deadline you are **required** to **upload your team’s submission** file on kaggle. The same deadline applies to the submission if the **presentation** together with **the full codes** . Please do it by email to the lecturer pwojcik@wne.uw.edu.pl or sm.lis@student.uw.edu.pl or share your Github account. The codes should **load the training data** , apply data transformation/selection, **train** the single best algorithm, **apply** this model on the **test data** and save test data predictions in the `csv` file. 

Separately share the codes which you applied to find the best algorithm, parameter search, etc. 

All teams will give presentations (not more than **10 minutes** ) informing about the algorithms considered, selection process and their **expected results** . 

Presentations will take place on 2026-05-29 for groups taught by Szymon Lis and Michał Woźniak and on 2026-06-01 for groups taught by Piotr Wójcik. 

Groups that do NOT present their results in class **will not be graded** . 

## Important dates again 

- 2026-04-20 by end of day - submission of information about the **team members** 

- 2026-05-28 by 23:55 - submission of **presentations** , **codes** and uploading **test sample predictions** on kaggle, 

- 2026-05-29 or 2026-06-01 - in class **presentations** - depending on the group. 

GOOD LUCK and happy modeling !!!!!!!!!!!!!!! 

