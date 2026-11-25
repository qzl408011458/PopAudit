# Independent recomputation of the lower block of Tab sigma-calibration from released-pipeline L2 outputs.
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd, numpy as np
B=_PA.R("/experiments/E_state/audits_v3")
rows=[]
for st in ["CA","TX","OH"]:
    for y in ["2018","2021","2024"]:
        d=pd.read_csv(f"{B}/layer2_semapop_vae_{st}_{y}/layer2_historical_sigma_summary.csv")
        s=d.sigma.values
        # check prior-independence
        d2=pd.read_csv(f"{B}/layer2_tvae_{st}_{y}/layer2_historical_sigma_summary.csv")
        same=np.allclose(d.sort_values('target').sigma.values,d2.sort_values('target').sigma.values)
        i=s.argmax()
        rows.append(dict(state=st,year=y,n=len(s),at_floor=int(np.isclose(s,0.005).sum()),
            median=round(float(np.median(s)),4),max=round(float(s.max()),3),argmax=d.target.iloc[i],
            l2=round(float(np.sqrt((s**2).sum())),3),prior_indep=same))
out=pd.DataFrame(rows); print(out.to_string()); out.to_csv(__file__.replace('.py','.csv'),index=False)
