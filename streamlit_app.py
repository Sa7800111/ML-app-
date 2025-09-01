import os, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
load_dotenv()

API = os.getenv("FASTAPI_PREDICT_URL", "http://localhost:8000/predict")
PAST = os.getenv("FASTAPI_PAST_PREDICTIONS_URL", "http://localhost:8000/past-predictions")

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")
st.title("Titanic — Predictor (5 features)")

tab1, tab2 = st.tabs(["Predict", "Past predictions"])

with tab1:
    st.subheader("Single prediction")
    c1,c2,c3 = st.columns(3)
    Pclass = c1.selectbox("Pclass", [1,2,3], index=2)
    Sex = c2.selectbox("Sex", ["male","female"])
    Embarked = c3.selectbox("Embarked", ["S","C","Q"])
    Age = st.number_input("Age", 0.0, 100.0, 30.0, 1.0)
    Fare = st.number_input("Fare", 0.0, 600.0, 7.25, 0.5)

    if st.button("Predict"):
        rec = {"Pclass":Pclass,"Sex":Sex,"Age":Age,"Fare":Fare,"Embarked":Embarked}
        r = requests.post(API, json={"records":[rec], "source":"webapp"}, timeout=20)
        st.write(r.json())

    st.divider()
    st.subheader("Batch prediction (CSV)")
    up = st.file_uploader("Upload CSV with columns: Pclass, Sex, Age, Fare, Embarked", type=["csv"])
    if st.button("Predict CSV") and up is not None:
        df = pd.read_csv(up)
        r = requests.post(API, json={"records": df.to_dict("records"), "source":"webapp"}, timeout=60)
        res = r.json()
        if "predictions" in res:
            out = df.copy(); out["pred"] = res["predictions"]
            if res.get("probabilities"): out["proba_1"] = res["probabilities"]
            st.dataframe(out)
        else:
            st.error(res)

with tab2:
    st.subheader("Past predictions")
    source = st.selectbox("Source", ["all","webapp","scheduled"], 0)
    limit = st.slider("Rows", 10, 200, 50, 10)
    r = requests.get(PAST, params={"limit":limit, "source":source}, timeout=20)
    st.dataframe(pd.json_normalize(r.json().get("items", [])))
