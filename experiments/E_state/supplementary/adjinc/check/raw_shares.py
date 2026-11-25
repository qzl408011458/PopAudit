"""Independent check: read ADJINC/HINCP straight from raw PUMS (not the parquet extracts),
merge on production person files, recompute PWGTP shares nominal vs constant-2013 (2013-file scale)."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[5])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import zipfile, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(_PA.R("")); A=Path(_PA.R("/experiments/E_state/supplementary/adjinc"))
OUT=A/"check"
CPI={2009:214.537,2010:218.056,2011:224.939,2012:229.594,2013:232.957,2018:251.107,2021:270.970,2024:313.689}
FIPS={"CA":6,"OH":39,"TX":48}
def bins(x):
    x=pd.to_numeric(x,errors="coerce").fillna(0).to_numpy()
    return np.select([x<0,x==0,x<=25000,x<=50000,x<=100000],["Loss","None","Low","LM","UM"],"High")
hh={}
for y in [2009,2010,2011,2012]:
    for st in FIPS:
        z=zipfile.ZipFile(A/f"raw/h{st.lower()}_{y}.zip"); n=[k for k in z.namelist() if k.endswith(".csv")][0]
        hh[(st,y)]=pd.read_csv(z.open(n),usecols=["SERIALNO","ADJINC","HINCP"],dtype={"SERIALNO":str})
for st in FIPS:
    hh[(st,2013)]=pd.read_csv(ROOT/f"data/pums13/housing_2013_{st}_full.csv",usecols=["SERIALNO","ADJINC","HINCP"],dtype={"SERIALNO":str})
for y in [2018,2021,2024]:
    sc="STATE" if y>=2023 else "ST"; parts=[]
    for f in ["psam_husa.csv","psam_husb.csv"]:
        for ch in pd.read_csv(ROOT/f"data/pums{str(y)[2:]}/{f}",usecols=["SERIALNO","ADJINC","HINCP",sc],dtype={"SERIALNO":str},chunksize=400000):
            parts.append(ch[ch[sc].isin(FIPS.values())])
    d=pd.concat(parts)
    for st,f in FIPS.items(): hh[(st,y)]=d[d[sc]==f]
rows=[]
for (st,y),h in sorted(hh.items()):
    adj=h.ADJINC.dropna().unique()
    a13=hh[(st,2013)].ADJINC.dropna().unique()[0]
    p=ROOT/f"data/pums_states/{st}"/(f"historical/person_level_{st}_{y}.csv" if y<=2013 else f"reference_{y}_{st}.csv")
    df=pd.read_csv(p,usecols=["Person_ID","PWGTP","Household_Income_Class"])
    ser=df.Person_ID.astype(str).str.rsplit("-",n=1).str[0]
    hm=h.drop_duplicates("SERIALNO").set_index("SERIALNO")
    hinc=ser.map(hm.HINCP); adjp=ser.map(hm.ADJINC)
    map_={"Loss":"Loss_or_Negative","None":"No_Income","Low":"Low","LM":"Lower_Middle","UM":"Upper_Middle","High":"High"}
    nom=pd.Series(bins(hinc)).map(map_); 
    f=(adjp/a13)*CPI[2013]/CPI[y]
    real=pd.Series(bins(hinc*f)).map(map_)
    adjl=pd.Series(bins(hinc*adjp/1e6)).map(map_)
    w=df.PWGTP.to_numpy(float)
    def sh(l,c): return 100*w[(l==c).to_numpy()].sum()/w.sum()
    rows.append(dict(st=st,y=y,adjinc=",".join(map(str,adj)),n=len(df),unmatched=int((~ser.isin(hm.index)).sum()),
        agree_nom=(nom.values==df.Household_Income_Class.values).mean(),agree_adj=(adjl.values==df.Household_Income_Class.values).mean(),
        High_nom=sh(nom,"High"),High_real=sh(real,"High"),Low_nom=sh(nom,"Low"),Low_real=sh(real,"Low"),
        High_stored=sh(df.Household_Income_Class,"High"),fac=float(f.dropna().iloc[0])))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/"raw_shares.csv",index=False)
