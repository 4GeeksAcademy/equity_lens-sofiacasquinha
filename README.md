“Prior work on salary prediction (e.g. Düzgün et al., 2025) shows that ensemble models such as Random Forest can reach very high predictive accuracy, but at the cost of lower interpretability. In contrast, linear and Ridge regression offer lower error but much higher transparency. Because EquityLens is an audit tool meant to support fairness discussions, not a black-box optimizer, we follow that literature and adopt a regularised linear model (Ridge) as our wage benchmark.”


More complex models (RF, CatBoost, DNNs) could reduce predictive error, but would conflict with the project’s need for explainability and alignment with the wage-equation tradition (Blau & Kahn, OLS).

# EquityLens (US) – Gender Pay Gap Audit & Wage Benchmark

EquityLens is a diagnostic tool that helps organisations assess whether women are being paid fairly, and plan how to close the gap with a realistic budget.

It uses the US Current Population Survey (CPS) as a public proxy for “market pay” and learns a **gender-blind wage benchmark**: what a worker would typically earn in the labour market given their job characteristics, **without using gender as an input**.

---

## 1. Business problem

Companies increasingly want to know whether they have a gender pay gap, but they hit two classic obstacles:

1. **Naïve internal comparisons are misleading**  
   Raw differences mix many things at once: occupation, industry, experience, hours, union status, etc. Comparing two people without controlling for these factors confuses job composition with true inequality.

2. **They lack a neutral market benchmark**  
   Even if a woman earns less than a man in the same team, it is not obvious whether both are above or below what the labour market would pay for that job.

**EquityLens** addresses both issues:

- It first quantifies the **adjusted gender pay gap** using a Blau & Kahn–style framework.
- It then builds a **gender-blind wage model** that estimates neutral benchmark wages based only on job and worker characteristics.

These benchmarks can be used to:
- Audit current salaries against a neutral reference, and  
- Simulate remedial scenarios under budget constraints (future extension).

---

## 2. Data

- Source: **US Current Population Survey (CPS)**, microdata extract.  
- Size: ~300k individuals, 7 survey years.  
- Storage: loaded into a local **SQLite** database for reproducibility.

> Note: The raw CPS CSV and the SQLite database are large (>100 MB) and are therefore **not included** in this repository. To fully reproduce the pipeline you must obtain the CPS file separately and place it under `data/`.

Key variables used:

- Wages: `realhrwage` (real hourly wage), `lnrwg` (log real hourly wage).
- Demographics: `sex`, `age`, `race`, `marst`.
- Education & human capital: `educ99`, `ba`, `adv`, `potexp`, `potexp2`.
- Job characteristics: `uhrswork`, `annhrs`, `classwkr`, `union`, `ft`.
- Time: `year`.
- Aggregated occupation dummies (22 groups).
- Aggregated industry dummies (15 groups).

---

## 3. Methodology overview

The project is structured in four notebooks plus a Streamlit app.

### Notebook 1 – Data acquisition & SQL loading (`1_sql_build.ipynb`)

- Loads the CPS CSV into pandas.
- Renames conflicting columns (`Transport` → `transport_ind`, `transport` → `transport_occ`).
- Creates/opens the SQLite database at `data/sql_cps_database.db`.
- Writes the full dataset as a raw table `cps_raw`.
- Runs sanity queries to confirm successful ingestion.

This notebook is pure ingestion. No cleaning, filtering, or modelling happens here.

---

### Notebook 2 – EDA & cohort definition (`2_eda_and_stats.ipynb`)

Goals:

1. **Audit data quality** (missingness, impossible values, redundancies).
2. **Document the raw gender pay gap** across years, age, education, occupations, and industries.
3. **Define a clean modelling cohort** and a **predictor set**.

Key analytical findings:

1. **The gender gap exists before any controls**

   - Women earn roughly **32% less** per hour than men in real terms.  
   - The gap appears in every year, age band, education level, occupation, and industry.  
   - There is no subgroup where the gap disappears.  
   → The phenomenon is structural, not a local anomaly.

2. **Occupational and industrial segregation is extreme**

   - Men are heavily concentrated in construction, transport, and industrial trades.  
   - Women dominate healthcare, education, personal services, and office administration.  
   - High-pay sectors (finance, durables) are male-dominated; lower-pay sectors concentrate women.  
   → A significant part of the **raw** gap comes from where men and women work, not only from differences within the same job.

3. **Education does not solve the gap**

   - Educational attainment is very similar by sex.  
   - At advanced degrees (master, PhD, professional), women still earn ≈ **41% less** than comparable men.  
   → Higher education raises wages for everyone but does **not** close the gender gap; at the top it often widens.

Cohort & feature decisions:

- **Include** (determinants of wages):
  - Education and degrees (`educ99`, `ba`, `adv`).
  - Experience (`potexp`, `potexp2`) and age.
  - Hours and job structure (`uhrswork`, `annhrs`, `classwkr`, `union`, `ft`).
  - Occupation and industry dummies (22 + 15).
  - Year.

- **Exclude** (to avoid embedding discrimination or noise):
  - `race`, `marst` from the models (used descriptively, not as predictors).
  - `citizen`, `nativity` (too much non-random missingness).
  - Redundant/noisy wage and schooling variables (`incwage`, `hrwage`, `sch`).

The final cohort is implemented in SQL as the view `vw_model_cohort` (see `sql/01_load_curate.sql`):

- Age 25–64, prime-age workers.
- Valid log wages (`lnrwg > 0`).
- Reasonable hours (`uhrswork` and `annhrs` consistent).
- Exactly one occupation and one industry group flagged.
- Core human capital variables present.
- Latest year reserved for testing.

---

### Notebook 3 – Gender pay gap inference (`3_gender_gap_inference.ipynb`)

Objective: quantify the **adjusted gender pay gap** using a simplified **Blau & Kahn–style** framework.

Two OLS models on log wages (`lnrwg`):

1. **Model A – job factors only (no gender)**  
   - Controls: education, experience, occupation, industry, hours, union, full-time, class of worker, etc.  
   - R² ≈ **0.40**.  
   - Interpreted as: how much of wage variation is explained by the composition of jobs and worker characteristics, without gender.

2. **Model B – job factors + gender (`female`)**  
   - Same controls as Model A plus a dummy `female`.  
   - The coefficient on `female` gives the **adjusted wage gap**.  
   - Result: coefficient ≈ **−0.21**, implying women earn about **19% less** even when holding job factors constant.  
   - R² increases by ≈ **2 percentage points**, showing that gender adds real explanatory power beyond job characteristics.

Robustness:

- A simple Ridge regression with standardisation produces almost the **same** female coefficient, confirming the effect is not driven by a single dummy or multicollinearity in the design matrix.

Fairness residual analysis:

- **Model A (no gender)**:
  - Overpredicts wages for men.
  - Underpredicts wages for women.
- **Model B (with gender)**:
  - Residuals for both groups move close to zero and become symmetric.

This pattern confirms that the adjusted gap is a stable feature of the data, not a modelling artefact.

All key metrics are saved to `artifacts/inference_metrics.json`.

---

### Notebook 4 – Gender-blind wage benchmark (`4_ml_model.ipynb`)

Objective: build a **gender-blind** wage model that estimates “neutral” market wages based on job characteristics only, to be used by the Streamlit app.

Setup:

- Data: same `vw_model_cohort` as Notebook 3.
- Target: `lnrwg` (log real hourly wage).
- Features: all job and worker characteristics **excluding** `sex`.
- Temporal split: all pre-2013 years for training; **2013** as test.

Models:

1. **Linear Regression (baseline)**  
   - Test R² ≈ **0.414**, MAE ≈ **0.364**.  
   - Train and test metrics are almost identical → no clear overfitting.

2. **Ridge + StandardScaler (final model)**  
   - Grid search over `alpha` in [0.01, 0.1, 1, 10, 100, 1000].  
   - Best `alpha = 100`.  
   - Test R² and MAE are essentially identical to plain Linear Regression.  
   - Ridge is chosen for deployment because it keeps coefficients stable in a high-dimensional dummy space, without sacrificing accuracy.

3. **Random Forest (sanity check)**  
   - Fitted on a 20k subsample.  
   - Strong overfitting (train R² ≈ 0.86) and **worse** test performance than Ridge.  
   → Confirms that the wage structure in this cohort is close to linear, and a linear model is preferable for a transparent benchmark.

Fairness residuals (gender-blind Ridge):

- The model systematically:
  - Overpredicts for men (positive mean residual),
  - Underpredicts for women (negative mean residual),
  - With similar absolute errors for both groups.

This is exactly what we expect from a **gender-blind** benchmark in a labour market where a real gender pay gap exists.

Artifacts written for deployment:

- `artifacts/ridge_model.pkl` – `StandardScaler + Ridge` pipeline.
- `artifacts/feature_list.json` – ordered list of feature names expected by the model.
- `artifacts/model_metrics.json` – performance and fairness summary for display in the UI.

---

## 4. Streamlit app

The app is implemented in `streamlit_app.py` and has two main workflows:

1. **CSV upload – batch audit**

   - Upload a CSV with one row per worker.
   - The app:
     - Drops columns not used by the model,
     - Adds missing feature columns with 0,
     - Reorders columns to match `feature_list.json`,
     - Uses the Ridge pipeline to predict `lnrwg` and converts it to hourly wages.
   - Outputs:
     - A preview of the input data,
     - Benchmark hourly wage per worker,
     - Summary statistics,
     - Downloadable CSV with predictions.

2. **Single-worker benchmark – manual test**

   - User enters age, weekly hours and optionally education, union status, occupation, industry, etc.
   - The app builds a single-row feature vector and predicts:
     - The gender-blind benchmark hourly wage.
   - Optionally, if the actual hourly wage is entered, the app computes the absolute and percentage gap vs. the benchmark.

The model **never** uses gender as an input. The benchmark is gender-blind by construction.

---

## 5. Project structure

Main elements:

- `data/`
  - `CurrentPopulationSurvey.csv` (not in repo; required locally).
  - `sql_cps_database.db` (SQLite database; not in repo if size-restricted).
- `sql/`
  - `01_load_curate.sql` – creates and populates `vw_model_cohort`.
- `notebooks/`
  - `1_sql_build.ipynb`
  - `2_eda_and_stats.ipynb`
  - `3_gender_gap_inference.ipynb`
  - `4_ml_model.ipynb`
- `artifacts/`
  - `ridge_model.pkl`
  - `feature_list.json`
  - `model_metrics.json`
  - `inference_metrics.json`
- `streamlit_app.py`
- `requirements.txt`
- `README.md` (this file)
