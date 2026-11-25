"""kappa re-check on the v3 certificate.
(a) fine kappa_2 ladder 3.5..7.0 step 0.25: #VALID and the cells that enter at each step;
(b) kappa_1,inv (invalid-mass tolerance) ladder: #VALID, per-model valid, verdict;
(c) B0 ladder on the POST-projection d_A decision per scenario;
(d) knife-edge cells: |chi_H - cutoff| / cutoff < 15% and |invalid - 0.005| / 0.005 < 50%.
Gates identical to decision_certificate_xstate_v3.py (L2 cutoff = kappa_2 + eta2 read back from L2_cutoff - 5.0)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import pandas as pd, numpy as np
from pathlib import Path
V=Path(_PA.R('/experiments/E_state/audits_v3/PAPER_xstate_v3')); OUT=V/'sensitivity'
d=pd.read_csv(V/'xstate_certificate_all.csv'); d=d[d.year.notna()].copy(); d['year']=d.year.astype(int)
d['eta2']=d.L2_cutoff-5.0
N={'CA':330109,'OH':64112,'TX':212756}
def dkw(a,n,k=2): return float(np.sqrt(np.log(2*max(k-1,1)/a)/(2*n)))
def l1_accept(r,k_inv=0.005,alpha=0.05):
    if r.L1_invalid>k_inv: return False
    a=alpha/3; n=N[r.state]; K=50
    return (r.L1_marg<=0.05-dkw(a,K*n,10)) and (r.L1_dep<=0.10-dkw(a,K*n,10)) and (1-r.L1_support<=0.30-dkw(a,n,1))
def valid(r,k2=5.0,k_inv=0.005):
    return l1_accept(r,k_inv) and (r.L2_chi_H<=k2+r.eta2) and bool(r.L3_accept)
lines=['# κ 敏感性复查（v3 证书，2026-09-02）','']
# (a)
lines+=['## (a) κ₂ 细梯度（3.5–7.0，步长 0.25）','','| κ₂ | #VALID | 2018/2021/2024 | 新进入的 cell |','|---|---|---|---|']
prev=set(); ra=[]
for k2 in np.arange(3.5,7.01,0.25):
    v=d[d.apply(lambda r:valid(r,k2),axis=1)]; cells=set(zip(v.state,v.model,v.year)); new=sorted(cells-prev)
    byy={y:int((v.year==y).sum()) for y in (2018,2021,2024)}
    ra.append({'k2':k2,'n_valid':len(v),**{f'n_{y}':byy[y] for y in byy},'entering':'; '.join(f'{s}×{m}×{y}' for s,m,y in new)})
    lines.append(f"| {k2:.2f} | {len(v)} | {byy[2018]}/{byy[2021]}/{byy[2024]} | {', '.join(f'{s}×{m}×{y}' for s,m,y in new) or '–'} |"); prev=cells
pd.DataFrame(ra).to_csv(OUT/'kappa2_fine_ladder.csv',index=False)
# (b)
lines+=['','## (b) κ₁,inv（invalid-mass 容忍度）梯度，κ₂=5 固定','','| κ₁,inv | L1 拒 | #VALID | 按模型 valid | max R_2^MA (valid) | 判定 |','|---|---|---|---|---|---|']
rb=[]
for k in (0.005,0.0075,0.010,0.015,0.020,0.030):
    l1=d.apply(lambda r:l1_accept(r,k),axis=1); v=d[d.apply(lambda r:valid(r,5.0,k),axis=1)]
    pm=v.groupby('model').size().to_dict(); mx=v.R_2_MA.max() if len(v) else np.nan
    rb.append({'k_inv':k,'n_L1_reject':int((~l1).sum()),'n_valid':len(v),'per_model':pm,'max_R2MA':mx})
    lines.append(f"| {k} | {int((~l1).sum())} | {len(v)} | {', '.join(f'{m} {n}' for m,n in sorted(pm.items(),key=lambda x:-x[1]))} | {mx:.3f} | {'ADOPT' if mx<=0.30 else 'DEFER'} |")
pd.DataFrame(rb).to_csv(OUT/'kappa1_inv_ladder.csv',index=False)
# (c)
e=pd.read_csv(V/'figures/tables/tableE_decision_dA.csv')
lines+=['','## (c) B₀ 梯度（post-projection d_A 口径，κ 固定）','','| B₀ | CA 2018 | OH 2018 | TX 2018 | OH 2024 | TX 2024 | 可部署场景数 |','|---|---|---|---|---|---|---|']
Dd={(s,y):g.d_A.min() for (s,y),g in e.groupby(['state','year'])}
rc=[]
for b0 in (0.15,0.20,0.25,0.30,0.35,0.40):
    dec={k:('Deploy' if Dd[k]<=b0 else 'Defer') for k in Dd}; n=sum(x=='Deploy' for x in dec.values())
    rc.append({'B0':b0,**{f'{s}_{y}':dec[(s,y)] for (s,y) in Dd},'n_deploy':n})
    lines.append(f"| {b0:.2f} | "+" | ".join(dec[k] for k in [('CA',2018),('OH',2018),('TX',2018),('OH',2024),('TX',2024)])+f" | {n} |")
pd.DataFrame(rc).to_csv(OUT/'B0_ladder_postproj.csv',index=False)
lines.append(''); lines.append('D† per scenario: '+', '.join(f'{s} {y} = {v:.3f}' for (s,y),v in sorted(Dd.items(), key=lambda x:(x[0][1],x[0][0]))))
# (d)
lines+=['','## (d) 刀口 cell','','L2（|χ̄_H − cutoff|/cutoff < 15%，仅 L1 通过者）：','']
k=d[(d.L1_accept==True)&((d.L2_chi_H-d.L2_cutoff).abs()/d.L2_cutoff<0.15)]
for _,r in k.sort_values('L2_chi_H').iterrows(): lines.append(f"- {r.state}×{r.model}×{r.year}: χ̄_H {r.L2_chi_H:.3f} vs cutoff {r.L2_cutoff:.3f} → {'pass' if r.L2_accept else 'fail'}")
lines+=['','L1 invalid mass（0.5×–1.5× 容忍度 0.005）：','']
for _,r in d.drop_duplicates(['state','model']).iterrows():
    if 0.0025<r.L1_invalid<0.0075: lines.append(f"- {r.state}×{r.model}: invalid {r.L1_invalid:.4f} ({r.L1_invalid/0.005:.2f}×) → {'fail' if r.L1_invalid>0.005 else 'pass'}")
# 2026-11-24 (group-1 fix): the binding count was a hard-coded sentence (20 valid / 14 binding); it is now read
# from the Layer-3 outputs (max_a chi_a^proj >= 0.999 in layer3_per_realization_diagnostics.csv).
import json as _json
_AUD=V.parent; _vv=d[d.VALID==True]; _bind=[]; _free=[]
for _,r in _vv.iterrows():
    _pl=_json.loads(pd.read_csv(_AUD/f"layer3_paperW_{r.model}_{r.state}_{r.year}"/"layer3_per_realization_diagnostics.csv").attribute_projected_losses_json.iloc[0])
    (_bind if max(_pl.values())>=0.999 else _free).append(f"{r.state}×{r.model}×{r.year}")
lines+=['',f"L3（σ(H)）：τ_a=1.0 在 {len(_vv)} 个 valid cell 中的 {len(_bind)} 个处 binding（max_a χ_a^proj = 1.000），其余 {len(_free)} 个（{', '.join(_free) if _free else '无'}）r* 本身已满足容差；τ<1 时仍可重新求解且可行（KL 升、nESS 降，见 l3_stress）。"]
(OUT/'kappa_recheck.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
