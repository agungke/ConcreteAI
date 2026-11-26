import numpy as np, pandas as pd
from src.train import FEATURES,TARGET,split_by_mixture,regression_metrics

def test_group_split_prevents_mixture_leakage():
    rows=[]
    for mix in range(20):
        for age in [3,7,28]: rows.append([100+mix,20,0,180,2,900,800,age,10+mix+age/10])
    df=pd.DataFrame(rows,columns=FEATURES+[TARGET]); tr,te,groups=split_by_mixture(df)
    assert set(groups[tr]).isdisjoint(set(groups[te]))

def test_metrics_are_finite():
    scores=regression_metrics([1,2,3],[1,2,4])
    assert all(np.isfinite(list(scores.values()))) and scores["MAE"]>0
