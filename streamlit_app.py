import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

#artifacts

def load_artifacts():
    artifacts_dir = Path(__file__).parent / "artifacts"

    model_path = artifacts_dir / "ridge_model.pkl"
    feature_list_path = artifacts_dir / "feature_list.json"
    metrics_path = artifacts_dir / "model_metrics.json"

    model = joblib.load(model_path)

    with open(feature_list_path, "r") as f:
        feature_list = json.load(f)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return model, feature_list, metrics


#mappings

EDUC_LEVELS = {
    "": (None, 0, 0),
    "Less than high school": (5, 0, 0),
    "High school graduate": (9, 0, 0),
    "Some college / associate": (13, 0, 0),
    "Bachelor's degree": (16, 1, 0),
    "Advanced degree (MA/PhD)": (18, 0, 1),
}

CLASSWKR_OPTIONS = {
    "": None,
    "Private wage/salary": 21,
    "Government": 24,
    "Self-employed": 10,
    "Unpaid family worker": 29,
}

UNION_OPTIONS = {
    "": None,
    "No union coverage": 1,
    "Union member": 2,
    "Covered, not member": 3,
}

OCCUPATION_OPTIONS = {
    "": None,
    "Manager": "manager",
    "Business": "business",
    "Financial operations": "financialop",
    "Computer": "computer",
    "Architecture": "architect",
    "Science": "scientist",
    "Social work": "socialworker",
    "Post-secondary education": "postseceduc",
    "Legal education": "legaleduc",
    "Artist": "artist",
    "Lawyer / Physician": "lawyerphysician",
    "Healthcare": "healthcare",
    "Healthcare support": "healthsupport",
    "Protective services": "protective",
    "Food / Care": "foodcare",
    "Building": "building",
    "Sales": "sales",
    "Office / Admin": "officeadmin",
    "Farmer": "farmer",
    "Construction / Extraction / Installation": "constructextractinstall",
    "Production": "production",
    "Transport (occupation)": "transport_occ",
}

INDUSTRY_OPTIONS = {
    "": None,
    "Finance": "finance",
    "Medical": "medical",
    "Education": "education",
    "Public administration": "publicadmin",
    "Professional services": "professional",
    "Durable manufacturing": "durables",
    "Non-durable manufacturing": "nondurables",
    "Retail trade": "retailtrade",
    "Wholesale trade": "wholesaletrade",
    "Transport (industry)": "transport_ind",
    "Utilities": "utilities",
    "Communications": "communications",
    "Social / Arts / Other services": "socartother",
    "Hotels and restaurants": "hotelsrestaurants",
    "Agriculture": "agriculture",
    "Mining and construction": "miningconstruction",
}


#page config

st.set_page_config(
    page_title="EquityLens (US): Wage Benchmark",
    layout="wide"
)

st.title("EquityLens: Gender-blind wage benchmark (US CPS)")

st.markdown(
    "This app uses a **Ridge regression model** trained on US CPS data to estimate "
    "**gender-blind benchmark wages** based on job and worker characteristics "
    "(age, hours, education, occupation, industry, union, full-time status).\n\n"
    "**Gender is not used as input.**"
)

#model and snapshot

try:
    model, model_features, metrics = load_artifacts()
except FileNotFoundError:
    st.error(
        "Artifacts not found. Ensure the `artifacts/` directory with the model, "
        "feature list, and metrics is in the same folder as this app."
    )
    st.stop()
except Exception as e:
    st.error(f"Unexpected error when loading artifacts: {e}")
    st.stop()

st.subheader("Model snapshot")

col1, col2 = st.columns(2)
with col1:
    st.write("**Model type:**", metrics.get("model_type", "N/A"))
    st.write('**Target:** `lnrwg` (log of real hourly wage)')

    train_years_str = ", ".join(map(str, metrics.get("train_years", [])))

    st.write("**Train years:**", train_years_str)
    st.write("**Test year:**", metrics.get("test_year", "N/A"))

with col2:
    st.write("**Ridge test R²:**", round(metrics.get("test_r2", np.nan), 3))
    st.write("**Ridge test MAE (log wage):**", round(metrics.get("test_mae", np.nan), 3))

#upload

st.header("Upload data for benchmark wages")

st.markdown(
    "Upload a CSV with one row per worker. Columns should match the **feature list** "
    "shown in the schema below.\n\n"
    "Extra columns will be dropped. Missing columns will be filled with 0. "
    "The benchmark is always computed without using gender."
)

with st.expander("See required feature columns (model input schema)"):
    st.write("Your CSV should contain these feature names:")
    st.code("\n".join(model_features))

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="File should contain the same feature columns used in the model (or a subset)."
)

#prediction

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        st.stop()

    for col in ["Unnamed: 0", "index"]:
        if col in df_input.columns:
            df_input = df_input.drop(columns=[col])

    st.subheader("Preview of uploaded data")
    st.write(df_input.head())

    X_app = df_input.copy()

    extra_cols = [c for c in X_app.columns if c not in model_features]
    if extra_cols:
        st.info(
            f"Dropping {len(extra_cols)} columns not used by the model: "
            f"{', '.join(extra_cols[:15])}"
            + (" ..." if len(extra_cols) > 15 else "")
        )
        X_app = X_app.drop(columns=extra_cols)

    missing_cols = [c for c in model_features if c not in X_app.columns]
    if missing_cols:
        st.info(
            f"Adding {len(missing_cols)} missing feature columns as 0 "
            "(typically meaning 'category not present')."
        )
        for col in missing_cols:
            X_app[col] = 0

    X_app = X_app[model_features]

    try:
        lnrwg_pred = model.predict(X_app)
    except Exception as e:
        st.error(f"Model could not generate predictions: {e}")
        st.stop()

    wage_pred = np.exp(lnrwg_pred)

    df_output = df_input.copy()
    df_output["benchmark_hourly_wage"] = wage_pred

    st.subheader("Predicted benchmark wages (first 10 rows)")
    st.write(df_output.head(10))

    st.subheader("Summary of predicted wages")
    summary_df = pd.DataFrame(
        {
            "metric": ["mean", "median", "min", "max"],
            "benchmark_hourly_wage": [
                float(np.mean(wage_pred)),
                float(np.median(wage_pred)),
                float(np.min(wage_pred)),
                float(np.max(wage_pred)),
            ],
        }
    )
    st.write(summary_df)

    st.subheader("Download results")
    csv_out = df_output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV with predictions",
        data=csv_out,
        file_name="equitylens_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("No file uploaded yet. Upload a CSV to compute benchmark wages.")


#single worker benchmark

st.header("Single worker benchmark")

st.markdown(
    "Use this section to test the model on a **single worker**. "
    "Only age and hours are required, all other fields are optional. "
    "All wages here are **hourly**."
)

with st.form("single_worker_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)

        uhrswork = st.number_input(
            "Usual weekly hours (uhrswork)",
            min_value=1,
            max_value=80,
            value=40,
            step=1,
        )

        ft_flag = st.checkbox("Full-time worker (ft = 1)", value=True)

        current_hourly_wage = st.number_input(
            "Current hourly wage (optional)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Enter the worker's actual hourly wage if you want to see the gap vs benchmark.",
        )

    with col_b:
        educ_label = st.selectbox(
            "Education level (educ99, optional)",
            list(EDUC_LEVELS.keys()),
            index=0,
        )

        classwkr_label = st.selectbox(
            "Class of worker (optional)",
            list(CLASSWKR_OPTIONS.keys()),
            index=0,
        )

        union_label = st.selectbox(
            "Union status (optional)",
            list(UNION_OPTIONS.keys()),
            index=0,
        )

    occ_label = st.selectbox(
        "Occupation group (optional, one per worker)",
        list(OCCUPATION_OPTIONS.keys()),
        index=0,
    )

    ind_label = st.selectbox(
        "Industry group (optional, one per worker)",
        list(INDUSTRY_OPTIONS.keys()),
        index=0,
    )

    submitted_single = st.form_submit_button("Compute benchmark for this worker")


if submitted_single:
    single_features = pd.DataFrame(
        data=[np.zeros(len(model_features))],
        columns=model_features,
    )

    #core
    if "age" in single_features.columns:
        single_features.loc[0, "age"] = age
    if "uhrswork" in single_features.columns:
        single_features.loc[0, "uhrswork"] = uhrswork
    if "annhrs" in single_features.columns:
        single_features.loc[0, "annhrs"] = uhrswork * 52

    #education
    educ99_val, ba_val, adv_val = EDUC_LEVELS[educ_label]
    if educ99_val is not None and "educ99" in single_features.columns:
        single_features.loc[0, "educ99"] = educ99_val
    if "ba" in single_features.columns:
        single_features.loc[0, "ba"] = ba_val
    if "adv" in single_features.columns:
        single_features.loc[0, "adv"] = adv_val

    # potexp
    if "potexp" in single_features.columns and educ99_val is not None:
        potexp_val = max(age - educ99_val - 6, 0)
        single_features.loc[0, "potexp"] = potexp_val

    if "potexp2" in single_features.columns and educ99_val is not None:
        single_features.loc[0, "potexp2"] = potexp_val ** 2

    #cls wrkr/union
    classwkr_code = CLASSWKR_OPTIONS[classwkr_label]
    union_code = UNION_OPTIONS[union_label]

    if classwkr_code is not None and "classwkr" in single_features.columns:
        single_features.loc[0, "classwkr"] = classwkr_code
    if union_code is not None and "union" in single_features.columns:
        single_features.loc[0, "union"] = union_code

    #full-time
    if "ft" in single_features.columns:
        single_features.loc[0, "ft"] = 1 if ft_flag else 0

    #occupation dummy
    occ_dummy_col = OCCUPATION_OPTIONS[occ_label]
    if occ_dummy_col is not None and occ_dummy_col in single_features.columns:
        single_features.loc[0, occ_dummy_col] = 1

    #industry dummy
    ind_dummy_col = INDUSTRY_OPTIONS[ind_label]
    if ind_dummy_col is not None and ind_dummy_col in single_features.columns:
        single_features.loc[0, ind_dummy_col] = 1

    try:
        single_lnrwg_pred = float(model.predict(single_features)[0])
    except Exception as e:
        st.error(f"Model could not generate prediction for this worker: {e}")
        st.stop()

    single_wage_pred = float(np.exp(single_lnrwg_pred))

    result_data = {
        "age": age,
        "uhrswork": uhrswork,
        "annhrs_used": uhrswork * 52,
        "education_level": educ_label,
        "ba": ba_val,
        "adv": adv_val,
        "classwkr": classwkr_label,
        "union": union_label,
        "ft": int(ft_flag),
        "occupation_group": occ_label,
        "industry_group": ind_label,
        "pred_lnrwg": single_lnrwg_pred,
        "benchmark_hourly_wage": single_wage_pred,
    }

    if current_hourly_wage > 0:
        gap_abs = current_hourly_wage - single_wage_pred
        gap_pct = (current_hourly_wage / single_wage_pred - 1) * 100

        result_data["actual_hourly_wage"] = current_hourly_wage
        result_data["gap_abs_hourly"] = gap_abs
        result_data["gap_pct_hourly"] = gap_pct

        st.info(
            "Computed the gender-blind benchmark and the gap between current hourly "
            "wage and the benchmark for this worker."
        )
    else:
        st.info(
            "Computed the gender-blind benchmark for this worker. "
            "Enter a current hourly wage to see the gap."
        )

    st.subheader("Single worker result")
    st.write(pd.DataFrame([result_data]))
