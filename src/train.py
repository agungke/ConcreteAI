"""Reproducible ConcreteAI training, grouped validation, EDA and artifacts."""
from pathlib import Path
import argparse, json
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"concrete_strength.csv"; MODELS=ROOT/"models"; RESULTS=ROOT/"results"
SEED=42
FEATURES=["cement_kg_m3","slag_kg_m3","fly_ash_kg_m3","water_kg_m3","superplasticizer_kg_m3","coarse_aggregate_kg_m3","fine_aggregate_kg_m3","age_days"]
TARGET="strength_mpa"; MIXTURE_FEATURES=FEATURES[:-1]

def load_and_validate(path=DATA):
    df=pd.read_csv(path)
    if list(df.columns)!=FEATURES+[TARGET]: raise ValueError("Unexpected dataset columns")
    if df.shape!=(1030,9): raise ValueError(f"Expected 1030x9 dataset, got {df.shape}")
    df=df.apply(pd.to_numeric,errors="raise")
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all(): raise ValueError("Missing/non-finite values")
    if (df[MIXTURE_FEATURES]<0).any().any() or not df.age_days.between(1,365).all() or (df[TARGET]<=0).any():
        raise ValueError("Out-of-domain values")
    return df

def split_by_mixture(df):
    groups=pd.factorize(pd.MultiIndex.from_frame(df[MIXTURE_FEATURES]))[0]
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(df[FEATURES],df[TARGET],groups))
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    return tr,te,groups

def regression_metrics(y,p):
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p))),"R2":float(r2_score(y,p))}

def save_eda(df):
    RESULTS.mkdir(parents=True,exist_ok=True)
    df.describe().T.to_csv(RESULTS/"descriptive_statistics.csv")
    df.corr().to_csv(RESULTS/"correlation_matrix.csv")
    plt.figure(figsize=(10,8)); sns.heatmap(df.corr(),annot=True,fmt=".2f",cmap="coolwarm",center=0)
    plt.tight_layout(); plt.savefig(RESULTS/"correlation_heatmap.png",dpi=160); plt.close()
    df[FEATURES].hist(figsize=(14,10),bins=25); plt.tight_layout(); plt.savefig(RESULTS/"feature_distributions.png",dpi=160); plt.close()

def main(data_path=DATA):
    df=load_and_validate(data_path); MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True); save_eda(df)
    tr,te,groups=split_by_mixture(df); Xtr=df.iloc[tr][FEATURES]; Xte=df.iloc[te][FEATURES]
    ytr=df.iloc[tr][TARGET]; yte=df.iloc[te][TARGET]; gtr=groups[tr]
    models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=500,random_state=SEED,n_jobs=-1)}
    metrics={}; preds={}; cvout={}
    cv=GroupKFold(n_splits=5)
    for name,model in models.items():
        model.fit(Xtr,ytr); ptr=model.predict(Xtr); pte=model.predict(Xte)
        metrics[name]={"train":regression_metrics(ytr,ptr),"test":regression_metrics(yte,pte)}; preds[name]=pte
        scores=cross_validate(model,Xtr,ytr,groups=gtr,cv=cv,n_jobs=-1,scoring={"mae":"neg_mean_absolute_error","rmse":"neg_root_mean_squared_error","r2":"r2"})
        cvout[name]={"MAE_mean":float(-scores["test_mae"].mean()),"MAE_std":float(scores["test_mae"].std()),"RMSE_mean":float(-scores["test_rmse"].mean()),"R2_mean":float(scores["test_r2"].mean())}
        joblib.dump(model,MODELS/("linear_regression.joblib" if name=="Linear Regression" else "random_forest.joblib"))
    summary={"dataset_rows":len(df),"split":{"method":"GroupShuffleSplit by seven ingredient quantities","train_rows":len(tr),"test_rows":len(te),"train_mixtures":len(set(groups[tr])),"test_mixtures":len(set(groups[te])),"seed":SEED},"exact_duplicate_rows":int(df.duplicated().sum()),"repeated_mixture_age_rows":int(df.duplicated(subset=FEATURES).sum()),"unique_mixtures":int(len(set(groups))),"metrics":metrics,"group_cv":cvout}
    (RESULTS/"metrics.json").write_text(json.dumps(summary,indent=2))
    pd.DataFrame({"actual_mpa":yte.to_numpy(),**{k.lower().replace(" ","_")+"_predicted_mpa":v for k,v in preds.items()}}).to_csv(RESULTS/"test_predictions.csv",index=False)
    pd.DataFrame([{"Model":k,**v["test"]} for k,v in metrics.items()]).to_csv(RESULTS/"model_comparison.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    for k,p in preds.items(): ax[0].scatter(yte,p,alpha=.6,label=k)
    lo,hi=float(yte.min()),float(yte.max()); ax[0].plot([lo,hi],[lo,hi],"k--"); ax[0].set(xlabel="Actual (MPa)",ylabel="Predicted (MPa)",title="Actual vs Predicted"); ax[0].legend()
    best=min(metrics,key=lambda k:metrics[k]["test"]["RMSE"]); residual=yte.to_numpy()-preds[best]
    ax[1].scatter(preds[best],residual,alpha=.6); ax[1].axhline(0,color="black",ls="--"); ax[1].set(xlabel="Predicted (MPa)",ylabel="Residual (MPa)",title="Residuals — "+best)
    fig.tight_layout(); fig.savefig(RESULTS/"prediction_diagnostics.png",dpi=160); plt.close(fig)
    rf=models["Random Forest"]; fi=pd.DataFrame({"feature":FEATURES,"importance":rf.feature_importances_}).sort_values("importance",ascending=False)
    fi.to_csv(RESULTS/"feature_importance.csv",index=False)
    plt.figure(figsize=(9,5)); sns.barplot(data=fi,x="importance",y="feature"); plt.tight_layout(); plt.savefig(RESULTS/"feature_importance.png",dpi=160); plt.close()
    pd.DataFrame({"feature":FEATURES+["intercept"],"coefficient":list(models["Linear Regression"].coef_)+[models["Linear Regression"].intercept_]}).to_csv(RESULTS/"linear_regression_coefficients.csv",index=False)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=DATA); main(p.parse_args().data)
