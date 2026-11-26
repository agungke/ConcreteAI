from pathlib import Path
import json, joblib, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parent
FEATURES=["cement_kg_m3","slag_kg_m3","fly_ash_kg_m3","water_kg_m3","superplasticizer_kg_m3","coarse_aggregate_kg_m3","fine_aggregate_kg_m3","age_days"]
st.set_page_config(page_title="ConcreteAI",page_icon="🏗️",layout="wide")
st.title("🏗️ ConcreteAI"); st.caption("Concrete Compressive Strength Prediction")
choice=st.sidebar.selectbox("Model",["Random Forest","Linear Regression"])
model_file=ROOT/"models"/("random_forest.joblib" if choice=="Random Forest" else "linear_regression.joblib")
if not model_file.exists(): st.error("Model belum tersedia. Jalankan python src/train.py."); st.stop()
model=joblib.load(model_file)
labels=["Cement","Blast Furnace Slag","Fly Ash","Water","Superplasticizer","Coarse Aggregate","Fine Aggregate","Concrete Age"]
defaults=[350,0,0,190,5,1000,750,28]; units=["kg/m³"]*7+["days"]
cols=st.columns(4); values={}
for i,(label,key,default,unit) in enumerate(zip(labels,FEATURES,defaults,units)):
    with cols[i%4]: values[key]=st.number_input(f"{label} ({unit})",min_value=1.0 if i==7 else 0.0,max_value=365.0 if i==7 else 2000.0,value=float(default),step=1.0)
if st.button("Predict Compressive Strength",type="primary",use_container_width=True):
    prediction=float(model.predict(pd.DataFrame([values],columns=FEATURES))[0])
    st.metric("Predicted Compressive Strength",f"{prediction:.2f} MPa"); st.info(f"Model: {choice}")
path=ROOT/"results"/"metrics.json"
if path.exists():
    result=json.loads(path.read_text()); st.subheader("Held-out Test Evaluation")
    st.dataframe(pd.DataFrame([{"Model":k,**v["test"]} for k,v in result["metrics"].items()]).set_index("Model"),use_container_width=True)
    a,b=st.columns(2)
    for col,name in [(a,"prediction_diagnostics.png"),(b,"feature_importance.png")]:
        if (ROOT/"results"/name).exists(): col.image(str(ROOT/"results"/name),use_container_width=True)
st.warning("Academic prototype only. Not a substitute for laboratory testing, structural design or construction safety decisions.")
