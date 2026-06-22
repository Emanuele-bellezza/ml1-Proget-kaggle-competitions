## Machine Learning 1 

Data description Piotr Wójcik, Szymon Lis, Michał Woźniak academic year 2025/2026 

## Datasets 

The data is **exactly the same for all teams** . 

## `restaurants` for **classification** 

The dataset consists of restaurant records with the following features: 

- **restaurant_id** - unique obs identifier 

- **has_photo** - Presence of profile photo 

- **user_ratings_total** - Cumulative count of user ratings 

- **price_level** - Price tier, 1-4 scale (categorical!) 

- **category_top20** - Google-defined restaurant category; non-top 20 as “Other” 

- **type_meal_takeaway** - Offers takeaway service 

- **type_meal_delivery** - Offers meal delivery service 

- **type_bar** - Co-located bar operation 

- **type_cafe** - Co-located café operation 

- **type_night_club** - Co-located night club operation 

- **types_count** - Count of co-located operations 

- **hours_open** - Total weekly hours open 

- **hours_open_weekends** - Total hours open on weekends 

- **hours_open_workdays** - Total hours open on workdays 

- **days_evenings_only** - Number of days open in the evening hours only 

- **days_mornings_only** - Number of days open in the morning hours only 

- **weekends_only** - Open only on weekends 

- **workdays_only** - Open only on workdays 

- **rating_5** - Count of 5-star ratings 

- **rating_4** - Count of 4-star ratings 

- **rating_3** - Count of 3-star ratings 

- **rating_2** - Count of 2-star ratings 

- **rating_1** - Count of 1-star ratings 

- **rating_avg** - Average rating 

- **rating_std** - Standard deviation of ratings 

- **tagcat_services** - Number of service-related tags (Wi-Fi, parking, etc.) 

- **tagcat_amenities** - Number of amenity-related tags (outdoor seating, etc.) 

- **tagcat_atmosphere** - Number of atmosphere-related tags (romantic, casual, etc.) 

- **tagcat_offerings** - Number of food/drink offering tags (vegan options, etc.) 

- **tagcat_social_inclusivity** - Number of social inclusivity related tags (kid-friendly, etc.) 

- **tagcat_payment_options** - Number of payment options (credit card, mobile pay, etc.) 

- **place_age_days** - Age of the restaurant in days 

- **first_review_year_max** - Year of first review 

- **ratings_num_1m_prior** - Number of ratings 1 month prior to survey 

- **ratings_num_3m_prior** - Number of ratings 3 months prior to survey 

- **ratings_num_6m_prior** - Number of ratings 6 months prior to survey 

- **ratings_num_9m_prior** - Number of ratings 9 months prior to survey 

- **ratings_num_12m_prior** - Number of ratings 12 months prior to survey 

- **ratings_avg_1m_prior** - Average rating 1 month prior to survey 

- **ratings_avg_3m_prior** - Average rating 3 months prior to survey 

- **ratings_avg_6m_prior** - Average rating 6 months prior to survey 

- **ratings_avg_9m_prior** - Average rating 9 months prior to survey 

- **ratings_avg_12m_prior** - Average rating 12 months prior to survey 

- **lang_pl_count** - Count of reviews in the Polish language 

- **rating_pl** - Average rating from reviews in Polish 

- **rating_foreign** - Average rating from reviews in languages other than Polish 

- **foreign_lang_share** - Proportion of reviews in foreign languages 

- **rating_mean_pl** - Mean rating across all Polish reviews 

- **rating_mean_foreign** - Mean rating across all foreignlanguage reviews 

- **rating_mean_lang_ratio** - Ratio of mean rating in Polish to mean rating in foreign languages 

- **review_has_text_pct** - Percentage of reviews that include text 

- **review_length_avg** - Average length of review text 

- **review_length_std** - Standard deviation of review text lengths 

- **catch_restaurant_count_500m** - Number of competing restaurants within 500meters radius 

- **catch_rating_avg_500m** - Average rating of competing restaurants within 500meter radius 

- **catch_place_age_days_500m** - Average age of competing restaurants within 500-meters radius 

- **catch_restaurant_count_1000m** - Number of competing restaurants within 1000-meters radius 

- **catch_rating_avg_1000m** - Average rating of competing restaurants within 1000-meter radius 

- **catch_place_age_days_1000m** - Average age of competing restaurants within 1000-meters radius 

- **catch_restaurant_count_2000m** - Number of competing restaurants within 2000-meters radius 

- **catch_rating_avg_2000m** - Average rating of competing restaurants within 2000-meter radius 

- **catch_place_age_days_2000m** - Average age of competing restaurants within 2000-meters radius 

- **residents** - Total population in the 1 km2 census grid cell where the restaurant is located 

- **restaurant_count** - Total number of restaurants the 1km2 census grid cell where the restaurant is located 

- **restaurants_per_capita** - Number of restaurants per capita in the 1km2 census grid cell where the restaurant is located 

- **urbanization** - Degree of urbanization, based on GHS-SMOD layer 

- **poi_count_100m** - Number of Points of Interest within 100-meters radius of the restaurant 

- **poi_count_200m** - Number of Points of Interest within 200-meters radius of the restaurant 

- **poi_count_500m** - Number of Points of Interest within 500-meters radius of the restaurant 

- **poi_count_1000m** - Number of Points of Interest within 1000-meters radius of the restaurant 

- **poi_count_2000m** - Number of Points of Interest within 2000-meters radius of the restaurant 

- **gus_60297** - Number of tourist accommodations in the county 

- **gus_60528** - Business deregistrations per capita in the county 

- **gus_64428** - Average monthly income in the county 

- **gus_79214** - Unemployment rate in the county 

- **gus_148074** - Number of foreign tourists in the county 

- **gus_153354** - Number of businesses created in the county 

- **gus_153398** - Number of businesses deregistered in the county 

- **gus_399257** - Utilization of tourist accommodation in the county 

- **gus_458173** - Business count per capita in the county 

- **gus_1548707_ratio** - New business registrations per capita in the county 

- **gus_60528_ratio** - Business deregistrations per capita in the county 

- **gus_1548707_net_ratio** - New business registrations net of business deregistrations in the county 

- **gus_152173_ratio** - Hospitality business per capita in the county 

- **affiliated** - Is a restaurant a part of a chain or franchise with at least 5 restaurants 

- **status_closed** - Is a restaurant permanently closed? **DEPENDENT VARIABLE** only in the training sample 

## Files Provided 

- `restaurants_train.csv` – training data contains 33296 observations and 86 columns along with the target variable **status_closed** . 

- `restaurants` _test.csv` – test data contains 8325 observations and 85 columns **without** the target variable. 

`annual.pay.usd` for **regression** 

## Target Variable 

|Column|Description|
|---|---|
|`annual.pay.usd`|Annual total compensation in USD (salary + bonuses|
||+ perks, before taxes). This is the variable you need|
||to predict.|



## Feature Dictionary 

## Demographics & Career 

|Column|Type|Description|
|---|---|---|
|`region`|Categorical|Anonymous geographic region code|
|||(R01–R18).|
|`age.group`|Ordinal|Age bracket: “18-24”, “25-34”, “35-44”,|
|||“45-54”, “55+”.|
|`education`|Ordinal|Highest completed education level (from|
|||primary school to professional/doctoral|
|||degree).|
|`is.dev.professional`|Categorical|Whether the respondent is a professional|
|||developer or codes as part of other|
|||work/studies.|
|`employment.type`|Categorical|Primary employment status: Full-time,|
|||Part-time, Freelance/Self-employed,|
|||Student, Job-seeking.|
|`work.location`|Categorical|Remote, Hybrid, or In-person.|
|`dev.role`|Categorical|Primary developer role (e.g., back-end,|
|||front-end, full-stack, mobile, DevOps,|
|||data engineer, etc.).|
|`people.manager`|Categorical|Whether the respondent is an individual|
|||contributor or a people manager.|



|Column|Type|Description|
|---|---|---|
|`industry`|Categorical|Industry sector of the respondent’s|
|||employer.|
|Experience|||
|Column|Type|Description|
|`coding.years.total`|Numeric|Total years of coding experience|
|||(including education).|
|`coding.years.professi`|Numeric|Years of professional (paid) coding|
|`onal`||experience.|
|`experience.years`|Numeric|Total years of work experience (any|
|||field).|
|Company & Workplace|||
|Column|Type|Description|
|`company.size`|Ordinal|Number of employees at the|
|||respondent’s company (from|
|||freelancer/solo to 10,000+).|
|`tech.purchase.influen`|Ordinal|Level of influence over technology|
|`ce`||purchasing decisions at work.|
|`build.vs.buy`|Categorical|Preference for building custom solutions|
|||vs. buying ready-made products.|
|`cloud.hosting`|Categorical|How the company hosts its applications|
|||(cloud, on-premises, hybrid).|
|`first.help.source`|Categorical|Where the respondent goes first for|
|||technical help at work.|
|`daily.search.time`|Ordinal|Time spent daily searching for answers|
|||to technical problems.|
|`daily.answer.time`|Ordinal|Time spent daily answering technical|
|||questions from colleagues.|



## Technical Skills (Multi-Select) 

These columns contain **semicolon-separated lists** of technologies. For example: `"JavaScript;Python;TypeScript"` . You will need to parse these to create useful features (e.g., binary indicators for individual technologies, count of technologies used, etc.). 

|etc.).||
|---|---|
|Column|Description|
|`prog.languages`|Programming languages used professionally in the|
||past year.|
|`databases`|Database systems used in the past year.|
|`cloud.platforms`|Cloud platforms used in the past year.|



Column Description `web.frameworks` Web frameworks and technologies used in the past year. `other.tech` Other frameworks and libraries used in the past year. `dev.tools` Developer tools (build, CI/CD, package managers) used in the past year. `dev.environments` IDEs and code editors used regularly. `personal.os` Operating systems used for personal purposes. `work.os` Operating systems used for professional purposes. `project.mgmt.tools` Project management and documentation tools used. `comm.tools` Communication and collaboration tools used. AI Tool Adoption Column Type Description `ai.search.tools` Multi-select AI-powered search and developer tools used in the past year. `ai.tools.used` Multi-select Specific AI tool use cases (e.g., writing code, searching for answers). `uses.ai` Categorical Whether the respondent currently uses AI tools in development. `ai.sentiment` Ordinal Favorability toward AI tools in the development workflow (from “Very unfavorable” to “Very favorable”). `ai.trust` Ordinal Level of trust in AI tool output accuracy. `ai.complex.rating` Ordinal Rating of how well AI tools handle complex tasks. `ai.job.threat` Categorical Whether the respondent believes AI threatens their job. Learning & Side Projects Column Type Description `how.learned.coding` Multi-select How the respondent learned to code (bootcamp, university, self-taught, etc.). `side.coding` Multi-select Types of coding activities done outside of work (hobby, open source, freelance, etc.). Job Satisfaction Column Type Description `job.satisfaction` Numeric (0–10) Overall satisfaction with current professional role. 

## Important Notes 

1. **Missing values** : Many columns contain missing values (NaN). Handling these appropriately is part of the task. Consider whether missingness itself might be informative. 

2. **Multi-select columns** : Columns like `prog.languages` , `databases` , etc. contain semicolon-separated lists. You need to decide how to encode these – common approaches include: 

   - Binary indicator columns for the most frequent items 

   - Count of items selected 

   - TF-IDF or frequency-based encoding 

3. **Target variable distribution** : The salary distribution is right-skewed. Consider whether a log transformation of the target improves your model. 

4. **Outliers** : Some salary values may be unusually low or high. Investigate and decide how to handle them. 

5. **Ordinal variables** : Several features have a natural ordering (age, education, company size, experience). Think about whether to treat them as numeric or categorical. 

6. **Region effects** : The `region` column captures geographic differences in compensation. Different regions have substantially different salary levels. 

## Files Provided 

- `train.csv` – training data contains 2,512 observations and 41 columns along with the target variable **annual.pay.usd** . 

- `test.csv` – test data contains 628 observations and 41 columns (40 features + `id` ) **without** the target variable. 

- `sample_submission.csv` – sample submission file in the correct format. 

