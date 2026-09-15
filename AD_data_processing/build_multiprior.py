"""Build the K=2 history CSV for multi-timepoint ControlNet injection, based on the clean
patient-level psplit. Adds prior_image_path/prior_follow_up/has_prior and prior2_image_path/prior2_follow_up/has_prior2.
"""
import pandas as pd, numpy as np, os
import os as _os
D=_os.environ.get("DERIVED_DIR", _os.path.join(_os.environ.get("DATA_DIR","."),"derived"))
pairs=pd.read_csv(f"{D}/AD-Progression-pairs-dt15.csv")
allv=pd.read_csv(f"{D}/AD-Progression-All-abs.csv")
allv=allv.sort_values(["subject_id","follow_up"])
by={k:v for k,v in allv.groupby("subject_id")}
p1p,p1f,h1,p2p,p2f,h2=[],[],[],[],[],[]
for _,r in pairs.iterrows():
    g=by.get(r.subject_id)
    pri=g[g.follow_up < r.starting_follow_up].sort_values("follow_up",ascending=False) if g is not None else None
    if pri is None or len(pri)==0:
        p1p.append(r.starting_image_path); p1f.append(r.starting_follow_up); h1.append(0)
        p2p.append(r.starting_image_path); p2f.append(r.starting_follow_up); h2.append(0)
    elif len(pri)==1:
        p1p.append(pri.iloc[0].image_path); p1f.append(pri.iloc[0].follow_up); h1.append(1)
        p2p.append(pri.iloc[0].image_path); p2f.append(pri.iloc[0].follow_up); h2.append(0)
    else:
        p1p.append(pri.iloc[0].image_path); p1f.append(pri.iloc[0].follow_up); h1.append(1)
        p2p.append(pri.iloc[1].image_path); p2f.append(pri.iloc[1].follow_up); h2.append(1)
pairs["prior_image_path"]=p1p; pairs["prior_follow_up"]=p1f; pairs["has_prior"]=h1
pairs["prior2_image_path"]=p2p; pairs["prior2_follow_up"]=p2f; pairs["has_prior2"]=h2
pairs["prior_age"]=pairs["starting_age"]-(pairs["starting_follow_up"]-pairs["prior_follow_up"])/1200.0
out=f"{D}/AD-Progression-multiprior-dt15.csv"; pairs.to_csv(out,index=False)
n=len(pairs); ntr=int(0.8*n)
print(f"saved {out}  n={n}")
print(f"  has_prior  = {np.mean(h1):.3f}   has_prior2 = {np.mean(h2):.3f}")
print(f"  train has_prior2 {np.mean(h2[:ntr]):.3f} | test has_prior2 {np.mean(h2[ntr:]):.3f}")
# check that the latents exist
LAT=f"{D}/latents_ddp_ae10_0875"
def lp(p): return os.path.join(LAT,p.split('/')[-3],p.split('/')[-2],p.split('/')[-1].split('.')[0]+".npz")
miss=sum(0 if os.path.exists(lp(p)) else 1 for p in pairs.prior2_image_path.iloc[:500])
print(f"  prior2 latents missing (first 500) = {miss}")
