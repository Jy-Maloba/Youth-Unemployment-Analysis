"""
Streamlit app: youth unemployment in Sub-Saharan Africa (SDG 8: AI and jobs).

Run from the project root:   streamlit run interface/app.py

It uses the files created in the notebook:
    notebook/youth_unemployment_model.pkl     the saved model (Ridge regression, trained on all 47 countries)
    notebook/sub_saharan_data.csv             the clean country-year table (df_clean)
    notebook/sub_saharan_ai_readiness.csv     the AI readiness index (2023)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ----------------------------------------------------------------------------------------------
# Settings: where the files are (relative to this file, so it works from any folder)
# ----------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "notebook" / "youth_unemployment_model.pkl"
DATA_PATH = ROOT / "notebook" / "sub_saharan_data.csv"
AI_PATH = ROOT / "notebook" / "sub_saharan_ai_readiness.csv"

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"      # same colours as the notebook charts
OWN_VALUES = "(type my own values)"

st.set_page_config(page_title="Youth unemployment in Sub-Saharan Africa", layout="wide")


# ----------------------------------------------------------------------------------------------
# Loading (cached so the files are only read once)
# ----------------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)                      # the dictionary saved in the notebook (model + details)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    ai = pd.read_csv(AI_PATH)
    # Attach the AI readiness score: by 3-letter country code if the data has one, otherwise by country name
    code_col = next((c for c in ["country_code", "Country Code", "code"] if c in data.columns), None)
    if code_col and "CountryCode" in ai.columns:
        data["ai_total"] = data[code_col].map(ai.set_index("CountryCode")["Total"])
    else:
        data["ai_total"] = data["Country"].map(ai.set_index("Country")["Total"])
    return data, ai


try:
    bundle = load_model()
    data, ai = load_data()
except FileNotFoundError as err:
    st.error(f"Could not find a required file: {err.filename}\n\nCheck that the notebook folder contains the model (.pkl) and the two CSV files.")
    st.stop()

if bundle.get("sklearn_version") != sklearn.__version__:
    st.warning(f"The model was saved with scikit-learn {bundle.get('sklearn_version')} but this app runs {sklearn.__version__}. "
               "It will usually still work; if you see errors, install the same version.")

# Latest available values for every country (used to pre-fill the inputs)
latest = data.sort_values("Year").groupby("Country").tail(1).set_index("Country")


# ----------------------------------------------------------------------------------------------
# The prediction function: raw numbers in, predicted youth unemployment (%) out
# ----------------------------------------------------------------------------------------------
def predict(agriculture, industry, services, internet, electricity, population_millions, labour_force_millions, ai_score):
    row = pd.DataFrame([{
        "agriculture_employment": agriculture, "industry_employment": industry, "services_employment": services,
        "internet_usage_percent": internet, "electricity_access": electricity,
        "log_population": np.log10(population_millions * 1e6),            # the model was trained on log10 of these two
        "log_labor_force": np.log10(labour_force_millions * 1e6),
        "ai_total": np.nan if ai_score is None else ai_score}])           # a missing AI score is filled by the model itself
    return float(bundle["model"].predict(row[bundle["features"]])[0])


# ----------------------------------------------------------------------------------------------
# Pre-filling the inputs from a chosen country
# ----------------------------------------------------------------------------------------------
def clip(value, low, high):
    return float(min(max(value, low), high))


def fill_from_country(name):
    r = latest.loc[name]
    st.session_state.update(
        agri=clip(r["agriculture_employment"], 0, 100), industry=clip(r["industry_employment"], 0, 100),
        services=clip(r["services_employment"], 0, 100), internet=clip(r["internet_usage_percent"], 0, 100),
        electricity=clip(r["electricity_access"], 0, 100), population=clip(r["population"] / 1e6, 0.1, 300),
        labour=clip(r["labor_force"] / 1e6, 0.01, 150), use_ai=bool(pd.notna(r["ai_total"])),
        ai_score=clip(r["ai_total"] if pd.notna(r["ai_total"]) else 30, 10, 70))


def on_country_change():
    if st.session_state["country_pick"] != OWN_VALUES:
        fill_from_country(st.session_state["country_pick"])


if "agri" not in st.session_state:                                    # first visit: start with Kenya
    fill_from_country("Kenya" if "Kenya" in latest.index else latest.index[0])

countries = sorted(latest.index)

# ----------------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------------
st.title("Youth unemployment in Sub-Saharan Africa")
st.caption("SDG 8 (Decent Work and Economic Growth): what kind of country has high youth unemployment, and where does Kenya stand?")

tab_predict, tab_explore, tab_about = st.tabs(["Estimate for a country", "Explore the data", "About the model"])

# ---------------------------------- Tab 1: estimate ----------------------------------
with tab_predict:
    st.write("Describe a country below, or start from a real one, and the model estimates its youth unemployment (ages 15-24).")
    st.selectbox("Start from a country", [OWN_VALUES] + countries, index=countries.index("Kenya") + 1 if "Kenya" in countries else 0,
                 key="country_pick", on_change=on_country_change,
                 help="Fills the inputs with that country's latest values. You can then change any of them.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Inputs")
        st.markdown("**Share of jobs (%)**")
        c1, c2, c3 = st.columns(3)
        agri = c1.number_input("Agriculture", 0.0, 100.0, step=0.5, key="agri")
        industry = c2.number_input("Industry", 0.0, 100.0, step=0.5, key="industry")
        services = c3.number_input("Services", 0.0, 100.0, step=0.5, key="services")
        job_total = agri + industry + services
        if abs(job_total - 100) > 3:
            st.warning(f"The three job shares add up to {job_total:.0f}%. They should be close to 100%.")

        st.markdown("**Connectivity and size**")
        internet = st.slider("Internet use (% of people)", 0.0, 100.0, step=0.5, key="internet")
        electricity = st.slider("Access to electricity (% of people)", 0.0, 100.0, step=0.5, key="electricity")
        d1, d2 = st.columns(2)
        population = d1.number_input("Population (millions)", 0.1, 300.0, step=0.5, key="population")
        labour = d2.number_input("Labour force (millions)", 0.01, 150.0, step=0.5, key="labour")

        use_ai = st.checkbox("I know the AI readiness score (0-100)", key="use_ai",
                             help="If unticked, the model fills this in with the typical score of the countries it learned from.")
        ai_score = st.slider("AI readiness score (2023)", 10.0, 70.0, step=0.5, key="ai_score", disabled=not use_ai)

    estimate = predict(agri, industry, services, internet, electricity, population, labour, ai_score if use_ai else None)
    typical_error = bundle["cv_rmse"]
    region_median = float(data.loc[data["Year"] == data["Year"].max(), "youth_unemployment"].median())

    with right:
        st.subheader("Estimate")
        m1, m2 = st.columns(2)
        m1.metric("Estimated youth unemployment", f"{estimate:.1f}%")
        m2.metric(f"Regional median ({int(data['Year'].max())})", f"{region_median:.1f}%")

        compare = {"This estimate": estimate, "Regional median": region_median}
        picked = st.session_state["country_pick"]
        if picked != OWN_VALUES:
            actual = float(latest.loc[picked, "youth_unemployment"])
            compare[f"{picked} (actual)"] = actual
        st.bar_chart(pd.Series(compare, name="Youth unemployment (%)"), horizontal=True, color=BLUE)

        st.info(f"**How much to trust this:** on countries the model had never seen, its estimates were typically off by about "
                f"**{typical_error:.1f} points** (just guessing the average was off by {bundle['baseline_rmse']:.1f}). "
                "Treat it as a rough estimate of the kind of country, not a precise forecast.")
        if picked != OWN_VALUES:
            st.caption(f"{picked} was part of the data the model learned from, so how close the estimate is to the actual value "
                       "is not a fair test of the model. The error figure above is the honest one.")

# ---------------------------------- Tab 2: explore ----------------------------------
with tab_explore:
    st.write("How has youth unemployment moved over time in one country compared with the regional median?")
    focus = st.selectbox("Country", countries, index=countries.index("Kenya") if "Kenya" in countries else 0, key="focus")

    by_year = data.groupby("Year")["youth_unemployment"].median().rename("Regional median")
    own = data[data["Country"] == focus].set_index("Year")["youth_unemployment"].rename(focus)
    trend = pd.concat([by_year, own], axis=1)
    trend.index = trend.index.astype(str)                              # show 2000 rather than 2,000
    st.line_chart(trend, color=[GREY, ORANGE])

    last = int(data["Year"].max())
    rank_table = data[data["Year"] == last].set_index("Country")["youth_unemployment"].sort_values(ascending=False)
    k1, k2, k3 = st.columns(3)
    if focus in rank_table.index:
        k1.metric(f"{focus}, {last}", f"{rank_table[focus]:.1f}%")
        k2.metric("Rank (1 = highest)", f"{list(rank_table.index).index(focus) + 1} of {len(rank_table)}")
    else:
        k1.metric(f"{focus}", "no data for the latest year")
    ai_row = ai[ai["Country"] == focus]
    k3.metric("AI readiness (2023)", f"{ai_row['Total'].iloc[0]:.1f}" if len(ai_row) else "not in the index")

    with st.expander("Show the data for this country"):
        st.dataframe(data[data["Country"] == focus].drop(columns=["ai_total"]).reset_index(drop=True))

# ---------------------------------- Tab 3: about ----------------------------------
with tab_about:
    st.subheader("The model")
    st.write(f"**{bundle['model_name']}**, trained on {bundle['trained_on']}. It predicts youth unemployment (ages 15-24) from the share of jobs in "
             "agriculture, industry and services, internet use, electricity access, population, labour force and the AI readiness score.")
    a1, a2 = st.columns(2)
    a1.metric("Typical error on unseen countries", f"{bundle['cv_rmse']:.1f} points")
    a2.metric("Error when always guessing the average", f"{bundle['baseline_rmse']:.1f} points")
    st.markdown(
        "**How it was tested:** whole countries were held out (5 folds, repeated 3 times), so every score comes from countries "
        "the model had never seen.\n\n"
        "**What it can and cannot do**\n"
        "- It estimates the *level* of youth unemployment for a country with certain characteristics.\n"
        "- It does **not** show that AI creates, replaces or changes jobs. The data has one AI readiness score per country and the models "
        "did not follow change over time within a country.\n"
        "- It is built on 47 countries, so it is a rough guide. Extreme cases such as Eswatini and South Africa are predicted poorly."
    )
