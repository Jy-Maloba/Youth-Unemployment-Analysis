# AI and Jobs in Sub-Saharan Africa: Where Does Kenya Stand?

**SDG 8 (Decent Work and Economic Growth) · CRISP-DM data science project**

---

## 1. Project summary

This project asks how artificial intelligence may be affecting jobs in Kenya, organised around three ideas:

| Axis | Question |
|---|---|
| **Job replacement** | Is AI replacing work that people did before? |
| **Job enhancement** | Is AI making existing jobs more productive? |
| **Job emergence** | Is AI creating new kinds of jobs? |

Kenya alone has only about 35 rows of yearly data, which is too little for machine learning. The project therefore uses a **panel of 47 Sub-Saharan African countries (2000-2024)** and treats **youth unemployment (ages 15-24)** as the main outcome, with Kenya as the focus country.

The work follows the **CRISP-DM** method and compares three families of models: **supervised**, **unsupervised** and **deep learning**.

## 2. Key findings

- **Kenya** had 15.5% youth unemployment in 2024, about double the regional median (8.3%). It ranks 15th of 44 countries. It stayed near 7% until 2016, then rose; the rise began in 2017.
- **Between countries**, youth unemployment is higher where more jobs are in services and fewer in farming, and where internet use and electricity access are higher. **Within a country over time**, these links almost vanish (correlations between -0.06 and +0.08). Only 4% of all variation in youth unemployment is movement within countries; 96% is differences between countries.
- **K-means** found three types of country: farm-based and offline (20 countries, 6.8% youth unemployment), in transition (12, 16.6%; Kenya is here), and services-led and connected (12, 25.8%). Youth unemployment was not given to the algorithm.
- **Model comparison** (error on countries the model never saw; lower is better):

  | Model | Family | RMSE | R² |
  |---|---|---|---|
  | Ridge regression | Supervised | **11.55** | 0.32 |
  | XGBoost | Supervised | 11.71 | 0.30 |
  | Neural network (MLP) | Deep learning | 11.92 | 0.28 |
  | Random forest | Supervised | 11.99 | 0.27 |
  | K-means cluster average | Unsupervised | 12.05 | 0.26 |
  | Decision tree | Supervised | 12.60 | 0.19 |
  | Baseline (always predict the training average) | - | 14.38 | -0.05 |

  Ridge regression is the best and the simplest, but the top models are close together and their uncertainty ranges overlap.
- **Kenya is a different case from the average:** the best model overall (Ridge) is not the best for Kenya (XGBoost is), and no model follows the rise in Kenya's youth unemployment after 2016.
- **AI readiness** (Kenya scores 40.2, 6th in the region) goes with connectivity but only weakly with youth unemployment, and it did not improve any model.
- **Main conclusion:** the models learn *what kind of country has high youth unemployment*, not *what happens to a country when AI arrives*. The results **do not prove that AI causes job replacement, enhancement or emergence.** They show where the data stops being able to answer the question.

## 3. Data

Seven World Bank indicators for all countries, plus the AI readiness index. Place them in a `data` folder exactly as follows (the notebook reads these paths):

```
data/
├── population-count.csv
├── labor force/API_SL.TLF.TOTL.IN_DS2_en_csv_v2_33277.csv
├── Individuals using the internet/API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv
├── API_SL.UEM.1524.ZS_DS2_en_csv_v2_435859.csv      youth unemployment (15-24)
├── API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_439770.csv      access to electricity
├── API_SL.AGR.EMPL.ZS_DS2_en_csv_v2_439956.csv      employment in agriculture
├── API_SL.IND.EMPL.ZS_DS2_en_csv_v2_428177.csv      employment in industry
├── API_SL.SRV.EMPL.ZS_DS2_en_csv_v2_469012.csv      employment in services
├── API_NY.GDP.PCAP.CD_DS2_en_csv_v2_468810.csv      GDP per capita (dropped, too many gaps)
├── API_SE.TER.ENRR_DS2_en_csv_v2_470152.csv         tertiary enrolment (dropped, too many gaps)
└── AI_Readiness_Index_2023_SSA.csv                  AI readiness, one score per country (2023)
```

Sources: World Bank World Development Indicators; Government AI Readiness Index 2023 (Sub-Saharan Africa).

**Cleaning:** GDP per capita and tertiary enrolment were dropped (too many missing values) and rows with any missing value were removed. The final panel has **1,116 country-years, 47 countries, 2000-2024**. Seychelles is removed entirely (no employment data), and Eritrea, South Sudan and Sudan have no complete 2024 row, so the 2024 snapshot has **44 countries**. The panel is unbalanced (42-47 countries per year).

## 4. How the project follows CRISP-DM

| Phase | What was done |
|---|---|
| **1. Business understanding** | Defined the SDG 8 question, the three axes (replacement, enhancement, emergence), and the pivot from Kenya-only data to a regional panel. |
| **2. Data understanding** | Loaded and inspected the 11 files; coverage, missing values, units. |
| **3. Data preparation** | Merged the indicators into one country-year table (`df_clean`); added the AI readiness score; log-transformed population and labour force. |
| **EDA** | **Univariate** (distributions, skew, rankings), **bivariate** (Kenya against the regional median over time; correlations between and within countries; change from 2000 to 2024; AI readiness against connectivity and unemployment) and **multivariate** (correlation heat map; countries grouped by how farm-based they are). |
| **4. Modelling** | One baseline, one unsupervised model (K-means, used both to find country types and as a predictor), supervised models (Ridge, decision tree, random forest, XGBoost) and a deep learning model (neural network, MLP). |
| **5. Evaluation** | Errors by country, type of country and period; Kenya in depth; whether the models can follow change over time; scorecard and recommendations. |
| **6. Deployment** | A Power BI dashboard that visualises the analysis (see section 7). |

### Modelling design (why the results can be trusted)

- **Target:** `youth_unemployment`. **Inputs:** agriculture, industry and services employment shares, internet use, electricity access, log population, log labour force, and AI readiness.
- **Leave-countries-out cross-validation:** whole countries are held out, never single rows. Each country has about 25 near-identical rows, so splitting by row would let a model recognise a country it had already seen and make the scores look far better than they are.
- **Nested tuning:** the settings of each model are chosen inside the training countries only, then tested on unseen countries. This is repeated 3 times with 5 folds (15 test sets).
- **Large leaves for tree models:** because rows of one country are near-copies, trees with small leaves memorise countries. The grids force leaves of at least 25 rows.
- **Metrics:** RMSE, MAE and R², all on unseen countries, each compared with a baseline that always predicts the training average.

## 5. Power BI dashboard

A three-page dashboard visualises the analysis, built from `df_clean.csv` and `ai_readiness_index.csv` (exported from the notebook):

1. **Kenya in the region:** youth unemployment against the regional median over time, a map, rankings and change since 2000.
2. **What goes with youth unemployment:** the between-country and within-country links, the farm-to-services groups.
3. **AI readiness:** Kenya's score and rank, the three pillars, and readiness against connectivity and unemployment.

`DASHBOARD_GUIDE.md` has the setup steps, the DAX measures and the page layouts.

## 6. Limitations

- **Only 47 countries**, so every score is uncertain, and an R² of about 0.3 means most differences between countries remain unexplained.
- **Associations, not causes.** The models and correlations cannot show that AI changes jobs.
- **AI readiness is a single 2023 snapshot** per country, with no time dimension, so it cannot be used to study change.
- **Extreme countries** (Eswatini, South Africa, Liberia and others) are predicted badly, and the models pull predictions towards the average.
- **Recorded youth unemployment is low in some of the poorest countries**, partly a measurement effect: where most people farm or work informally, they count as employed.
- The models predict *current* youth unemployment, not unemployment in later years.

## 7. Possible next steps

- Add data with a time dimension for AI (for example yearly AI adoption, AI-exposure measures or ICT employment).
- Model year-to-year *changes* rather than levels.
- Look closely at the countries the models predict worst.

## 9. Acknowledgements

Data: World Bank (World Development Indicators) and the Government AI Readiness Index 2023.
