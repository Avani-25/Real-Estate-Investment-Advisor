import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

# =========================================================
# THEME (dark / light toggle in sidebar)
# =========================================================
st.sidebar.markdown("## 🏠 Investment Advisor")
dark = st.sidebar.toggle("🌙 Dark mode", value=True)

if dark:
    C = dict(bg="#0E1117", side="#161A23", card="#1E2330", text="#FAFAFA", sub="#A8B0C0",
             border="#2E3445", accent="#4F8BF9", input="#1E2330")
    PLOT_TEMPLATE = "plotly_dark"
else:
    C = dict(bg="#FFFFFF", side="#F0F2F6", card="#F7F8FB", text="#1F2937", sub="#5B6475",
             border="#D9DDE6", accent="#2563EB", input="#FFFFFF")
    PLOT_TEMPLATE = "plotly_white"

st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {C['bg']} !important; color: {C['text']} !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
    background-color: {C['side']} !important; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li,
.stApp span, .stApp div[data-testid="stMarkdownContainer"], [data-testid="stSidebar"] * {{
    color: {C['text']}; }}
.stApp [data-testid="stCaptionContainer"], .stApp small {{ color: {C['sub']} !important; }}
[data-testid="stMetric"] {{ background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 14px 16px; }}
[data-testid="stMetricLabel"] p {{ color: {C['sub']} !important; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"],
.stNumberInput input, .stTextInput input, textarea {{
    background-color: {C['input']} !important; color: {C['text']} !important;
    border-color: {C['border']} !important; }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] {{
    background-color: {C['input']} !important; color: {C['text']} !important; }}
.stButton > button, .stDownloadButton > button {{
    border: 1px solid {C['accent']}; border-radius: 10px; }}
.stButton > button[kind="primary"] {{ background-color: {C['accent']}; color: #FFFFFF !important; }}
.stButton > button[kind="primary"] * {{ color: #FFFFFF !important; }}
hr {{ border-color: {C['border']} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 8px 10px; border-radius: 8px; width: 100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {C['card']}; }}
</style>
""", unsafe_allow_html=True)


# ---------- small helpers ----------
def show(fig):
    """Render a plotly figure using the current theme."""
    fig.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(d):
    try:
        st.dataframe(d, width="stretch")
    except TypeError:
        st.dataframe(d, use_container_width=True)


# ---------- loaders ----------
@st.cache_data
def load_data():
    d = pd.read_csv(ROOT / "data" / "processed_housing.csv")
    d["PPSF"] = d["Price_per_SqFt_calc"]
    return d


@st.cache_resource
def load_models():
    m = ROOT / "models"
    return (joblib.load(m / "classifier.joblib"), joblib.load(m / "regressor.joblib"),
            joblib.load(m / "clf_columns.joblib"), joblib.load(m / "reg_columns.joblib"))


def get_importance(pipe, top=12):
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    vals = (model.feature_importances_ if hasattr(model, "feature_importances_")
            else np.abs(model.coef_).ravel())
    s = pd.Series(vals, index=[n.split("__", 1)[1] for n in names])
    return s.nlargest(top).sort_values()


try:
    df = load_data()
    clf, reg, clf_cols, reg_cols = load_models()
except Exception as e:
    st.error(f"Could not load data or models: {e}\n\nRun notebooks 01, 03 and 04 first.")
    st.stop()


# ---------- feature builder (mirrors notebook 01 feature engineering) ----------
def build_row(i):
    transport = {"Low": 1, "Medium": 2, "High": 3}[i["transport"]]
    school_score = i["schools"] / 10
    hospital_score = i["hospitals"] / 10
    infra = (school_score + hospital_score + transport / 3) / 3
    return {
        "BHK": i["bhk"], "Size_in_SqFt": i["size"], "Price_in_Lakhs": i["price"],
        "Price_per_SqFt_calc": i["price"] * 100000 / i["size"],
        "Age_of_Property": i["age"], "Floor_No": i["floor"], "Total_Floors": i["total_floors"],
        "Nearby_Schools": i["schools"], "Nearby_Hospitals": i["hospitals"],
        "Parking_Count": 1 if i["parking"] == "Yes" else 0,
        "Amenities_Count": max(len(i["amenities"]), 1),
        "Transport_Score": transport, "School_Density_Score": school_score,
        "Hospital_Score": hospital_score, "Infrastructure_Score": infra,
        "Floor_Ratio": i["floor"] / max(i["total_floors"], 1),
        "SqFt_per_BHK": i["size"] / max(i["bhk"], 1),
        "State": i["state"], "City": i["city"], "Property_Type": i["ptype"],
        "Furnished_Status": i["furnished"], "Public_Transport_Accessibility": i["transport"],
        "Security": i["security"], "Availability_Status": i["availability"],
        "Facing": i["facing"], "Owner_Type": i["owner"],
    }


# =========================================================
# PAGE 1: PREDICT
# =========================================================
def page_predict():
    st.title("🔮 Predict")
    st.caption("Enter property details to see if it's a good investment and its estimated price after 5 years.")

    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(df["State"].unique()))
        city = st.selectbox("City", sorted(df[df["State"] == state]["City"].unique()))
        ptype = st.selectbox("Property type", sorted(df["Property_Type"].unique()))
        bhk = st.slider("BHK", 1, 5, 3)
        size = st.number_input("Size (sq ft)", 200, 10000, 1500, step=50)
        price = st.number_input("Current price (Lakhs)", 5.0, 5000.0, 150.0, step=5.0)
    with c2:
        age = st.slider("Age of property (years)", 0, 40, 10)
        total_floors = st.slider("Total floors in building", 1, 30, 10)
        floor = st.slider("Floor number", 0, total_floors, min(3, total_floors))
        furnished = st.selectbox("Furnished status", sorted(df["Furnished_Status"].unique()))
        facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
        availability = st.selectbox("Availability", sorted(df["Availability_Status"].unique()),
                                    format_func=lambda x: x.replace("_", " "))
    with c3:
        schools = st.slider("Nearby schools", 1, 10, 5)
        hospitals = st.slider("Nearby hospitals", 1, 10, 5)
        transport = st.selectbox("Public transport accessibility", ["Low", "Medium", "High"], index=1)
        parking = st.radio("Parking space", ["Yes", "No"], horizontal=True)
        security = st.radio("Security", ["Yes", "No"], horizontal=True)
        owner = st.selectbox("Owner type", sorted(df["Owner_Type"].unique()))
        all_am = sorted({a for x in df["Amenities"].unique() for a in x.split(", ")})
        amenities = st.multiselect("Amenities", all_am, default=all_am[:2])

    if st.button("Analyse this property", type="primary"):
        inp = dict(state=state, city=city, ptype=ptype, bhk=bhk, size=size, price=price, age=age,
                   floor=floor, total_floors=total_floors, furnished=furnished, facing=facing,
                   availability=availability, schools=schools, hospitals=hospitals, transport=transport,
                   parking=parking, security=security, owner=owner, amenities=amenities)
        row = pd.DataFrame([build_row(inp)])

        proba = clf.predict_proba(row[clf_cols["num"] + clf_cols["cat"]])[0]
        is_good = proba[1] >= 0.5
        confidence = proba[1] if is_good else proba[0]
        future = float(reg.predict(row[reg_cols["num"] + reg_cols["cat"]])[0])
        gain = future - price
        cagr = (future / price) ** (1 / 5) - 1

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("### Is this a Good Investment?")
            if is_good:
                st.success("✅ **Yes, good investment**")
            else:
                st.error("❌ **No, not a good investment**")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.progress(float(confidence))
        with r2:
            st.markdown("### Estimated price after 5 years")
            st.metric("Future price", f"₹ {future:,.1f} Lakhs",
                      delta=f"+{gain:,.1f} Lakhs ({cagr * 100:.1f}% per year)")
            st.caption(f"Current price: ₹ {price:,.1f} Lakhs · Price per sq ft: ₹ {price * 1e5 / size:,.0f}")

        years = list(range(0, 6))
        path = pd.DataFrame({"Year": years,
                             "Projected price (Lakhs)": [price * (1 + cagr) ** y for y in years]})
        show(px.line(path, x="Year", y="Projected price (Lakhs)", markers=True,
                     title="5-year price projection"))

        f1, f2 = st.columns(2)
        with f1:
            show(px.bar(get_importance(clf), orientation="h",
                        title="What drives the investment classification",
                        labels={"value": "Importance", "index": ""}).update_layout(showlegend=False))
        with f2:
            show(px.bar(get_importance(reg), orientation="h",
                        title="What drives the price forecast",
                        labels={"value": "Importance", "index": ""}).update_layout(showlegend=False))


# =========================================================
# PAGE 2: EXPLORE PROPERTIES
# =========================================================
def page_explore():
    st.title("🔎 Explore Properties")
    st.caption("Filter the dataset by area, price, BHK and more.")

    f1, f2, f3 = st.columns(3)
    with f1:
        fs = st.multiselect("State", sorted(df["State"].unique()))
        cities_pool = df[df["State"].isin(fs)]["City"].unique() if fs else df["City"].unique()
        fc = st.multiselect("City", sorted(cities_pool))
    with f2:
        fb = st.multiselect("BHK", sorted(df["BHK"].unique()))
        ft = st.multiselect("Property type", sorted(df["Property_Type"].unique()))
    with f3:
        pmin, pmax = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
        smin, smax = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
        pr = st.slider("Price (Lakhs)", pmin, pmax, (pmin, pmax))
        sz = st.slider("Area (sq ft)", smin, smax, (smin, smax))
    only_good = st.checkbox("Show only Good Investments")

    m = df["Price_in_Lakhs"].between(*pr) & df["Size_in_SqFt"].between(*sz)
    if fs: m &= df["State"].isin(fs)
    if fc: m &= df["City"].isin(fc)
    if fb: m &= df["BHK"].isin(fb)
    if ft: m &= df["Property_Type"].isin(ft)
    if only_good: m &= df["Good_Investment"] == 1
    res = df[m]

    k1, k2, k3 = st.columns(3)
    k1.metric("Matching properties", f"{len(res):,}")
    k2.metric("Avg price (Lakhs)", f"{res['Price_in_Lakhs'].mean():,.1f}" if len(res) else "n/a")
    k3.metric("Good investments", f"{res['Good_Investment'].mean() * 100:.1f}%" if len(res) else "n/a")

    cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Availability_Status", "Good_Investment", "Future_Price_5Y"]
    show_df(res[cols].head(500))
    st.download_button("Download filtered data (CSV)", res[cols].to_csv(index=False),
                       "filtered_properties.csv")


# =========================================================
# PAGE 3: MARKET INSIGHTS
# =========================================================
def page_insights():
    st.title("📊 Market Insights")

    top_cities = df.groupby("City")["Price_in_Lakhs"].mean().nlargest(15).index
    heat = (df[df["City"].isin(top_cities)]
            .pivot_table(index="City", columns="Property_Type", values="PPSF", aggfunc="mean").round(0))
    show(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
                   title="Heatmap: avg price per sq ft (₹), top 15 cities by price"))

    a, b = st.columns(2)
    with a:
        city_sel = st.selectbox("City for trend chart", sorted(df["City"].unique()), key="trend_city")
        t = df[df["City"] == city_sel].copy()
        t["Build period"] = (t["Year_Built"] // 5) * 5
        trend = t.groupby("Build period")[["Price_in_Lakhs", "Future_Price_5Y"]].mean().reset_index()
        show(px.line(trend, x="Build period", y=["Price_in_Lakhs", "Future_Price_5Y"], markers=True,
                     title=f"Avg current vs 5-year price by construction period: {city_sel}"))
    with b:
        good = df.groupby("City")["Good_Investment"].mean().nlargest(15).mul(100).round(1).reset_index()
        show(px.bar(good, x="Good_Investment", y="City", orientation="h",
                    title="Share of Good Investments (%), top 15 cities")
             .update_layout(yaxis={"categoryorder": "total ascending"}))

    show(px.histogram(df.sample(min(len(df), 50000), random_state=1), x="Price_in_Lakhs", nbins=50,
                      title="Distribution of property prices (Lakhs)"))


# =========================================================
# PAGE 4: MODEL INFO
# =========================================================
def page_model():
    st.title("🧠 Model Info")
    st.markdown("""
**Classification** (Good Investment): XGBoost, trained on a rule-based label (price vs city median,
price per sq ft vs city median, BHK ≥ 3, ready-to-move, infrastructure score).

**Regression** (Price after 5 years): XGBoost, trained on a target simulated from city-based growth
rates plus an infrastructure adjustment.
""")
    st.info("The dataset has no real future prices or labels, so both targets are simulated from rules. "
            "High scores show the models learned those rules; the no-price-feature variants show "
            "performance from property attributes alone.")
    perf = pd.DataFrame({
        "Model": ["Classifier: all features", "Classifier: no price features",
                  "Regressor: all features", "Regressor: no price features"],
        "Accuracy / RMSE": ["Acc 0.9983", "Acc 0.7076", "RMSE 1.39", "RMSE 214.7"],
        "F1 / R²": ["F1 0.9984", "F1 0.7304", "R² 1.000", "R² 0.002"],
    })
    st.table(perf)
    c, r = st.columns(2)
    with c:
        show(px.bar(get_importance(clf, 15), orientation="h", title="Classifier feature importance")
             .update_layout(showlegend=False))
    with r:
        show(px.bar(get_importance(reg, 15), orientation="h", title="Regressor feature importance")
             .update_layout(showlegend=False))


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("---")
PAGES = {
    "🔮 Predict": page_predict,
    "🔎 Explore Properties": page_explore,
    "📊 Market Insights": page_insights,
    "🧠 Model Info": page_model,
}
choice = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} properties · {df['City'].nunique()} cities")

PAGES[choice]()  