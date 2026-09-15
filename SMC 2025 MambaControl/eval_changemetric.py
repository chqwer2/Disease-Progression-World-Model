"""Change-metric eval for the SDEdit variant of MambaControl.
Raw PSNR vs copy is the wrong lens (copy is unbeatable when the true change is smaller than
the reconstruction noise). The right lens = compute_change_metrics (detrended,
direction-aware): does the predicted change field match the real change field where change
actually happens.
Reports CHANGE_PCC / CHANGE_DICE / CHANGE_MAE for: the method, vs a copy floor (pred=base) and
a NOISE control (pred=base+gaussian). Higher PCC/DICE, lower MAE = better."""
import os, sys, numpy as np, torch, pandas as pd, torch.nn.functional as F

# ---- Metric code inlined so this script does not import the utils package, which
#      instantiates lpips.LPIPS() at module level (a heavy, failure-prone side effect).
#      Keep it identical to the shared implementation. ----
def _as_np(x):
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).squeeze()
def _detrend(field, brain, sigma, eps=1e-6):
    if sigma is None: return field
    if not np.isfinite(sigma): return field - field[brain].mean()
    from scipy.ndimage import gaussian_filter
    bf = brain.astype(np.float32)
    num = gaussian_filter(field * bf, sigma); den = gaussian_filter(bf, sigma)
    return field - num / (den + eps)
def compute_change_metrics(pred, target, input, mask=None, detrend_sigma=16.0,
                           tau=None, tau_floor=0.02, tau_rel=0.75, eps=1e-6):
    pred, target, input = _as_np(pred), _as_np(target), _as_np(input)
    if mask is not None: brain = _as_np(mask) > 0.5
    else: brain = (input > 0.05) | (target > 0.05)
    if brain.sum() == 0: brain = np.ones_like(input, dtype=bool)
    gt_full = _detrend(target - input, brain, detrend_sigma)
    pr_full = _detrend(pred   - input, brain, detrend_sigma)
    gt = gt_full[brain]; pr = pr_full[brain]; agt = np.abs(gt)
    if tau is None: tau = max(tau_floor, float(np.quantile(agt, tau_rel)))
    gt_up, gt_dn = (gt > tau), (gt < -tau); pr_up, pr_dn = (pr > tau), (pr < -tau)
    gt_ch = gt_up | gt_dn; n_change = int(gt_ch.sum())
    change_mae = float(np.abs(gt[gt_ch] - pr[gt_ch]).mean()) if n_change > 0 else float(np.abs(gt - pr).mean())
    def _dice(a, b):
        s = a.sum() + b.sum()
        if s == 0: return 1.0
        return float((2.0 * np.logical_and(a, b).sum()) / (s + eps))
    w_up, w_dn = gt_up.sum(), gt_dn.sum()
    change_dice = float((w_up*_dice(pr_up,gt_up)+w_dn*_dice(pr_dn,gt_dn))/(w_up+w_dn)) if (w_up+w_dn)>0 else 1.0
    active = gt_ch | pr_up | pr_dn
    if active.sum() > 1 and gt[active].std() > eps and pr[active].std() > eps:
        change_pcc = float(np.corrcoef(gt[active], pr[active])[0, 1])
    elif active.sum() == 0: change_pcc = 1.0
    else: change_pcc = 0.0
    return {'CHANGE_MAE': change_mae, 'CHANGE_DICE': change_dice, 'CHANGE_PCC': change_pcc}
from monai.networks.schedulers import DDIMScheduler
from mambacontrol import networks
from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder
DEV="cuda"
AE     =os.environ.get("MC_AE_CKPT",  "checkpoints/autoencoder.pth")
UNET   =os.environ.get("MC_UNET_CKPT","checkpoints/unet.pth")
CNET_V3=os.environ.get("MC_CNET_CKPT","checkpoints/controlnet.pth")
N=24; T_START=200; STEPS=100
valid=pd.read_csv(os.environ.get("MC_PAIRS_CSV","data/pairs.csv"))
valid=valid.iloc[int(0.8*len(valid)):].reset_index(drop=True)
ae=init_autoencoder(AE).to(DEV).float().eval()
for m in ae.modules():
    if hasattr(m,"norm_float16"): m.norm_float16=False
diff=networks.init_mamba_diffusion(UNET).to(DEV).eval()
cnet=networks.init_mamba_controlnet(); cnet.load_state_dict(torch.load(CNET_V3,map_location="cpu")); cnet=cnet.to(DEV).eval()
def load_latent(path):
    z=torch.from_numpy(np.load(path.replace(".nii.gz","_new_latent.npz").replace(".nii","_new_latent.npz"))["data"]).float()
    pads=[]
    for d in reversed(z.shape[1:]): pads+=[0,(-d)%4]
    return F.pad(z,pads)
sf=float(1.0/torch.std(load_latent(valid["followup_image_path"].iloc[0])))
def dec(latent): return ae.decode_stage_2_outputs(latent.float())[0,0].clamp(0,1)
def ctx_of(r): return torch.tensor([[[float(r['followup_age']),0.5,float(r['starting_diagnosis']),float(r['starting_cerebral_cortex']),float(r['starting_hippocampus']),float(r['starting_amygdala']),float(r['starting_cerebral_white_matter']),float(r['starting_lateral_ventricle'])]]],dtype=torch.float32).to(DEV)
sched=DDIMScheduler(num_train_timesteps=1000,schedule="scaled_linear_beta",beta_start=0.0015,beta_end=0.0205)
sched.set_timesteps(num_inference_steps=STEPS)

acc={"method":{"pcc":[],"dice":[],"mae":[]}, "copy":{"pcc":[],"dice":[],"mae":[]}, "noise":{"pcc":[],"dice":[],"mae":[]}}
def add(key,pred,real,base):
    m=compute_change_metrics(pred, real, base, mask=None, detrend_sigma=16.0)
    for k,mk in (("pcc","CHANGE_PCC"),("dice","CHANGE_DICE"),("mae","CHANGE_MAE")):
        v=m.get(mk, m.get(mk.lower()))
        if v is not None and np.isfinite(v): acc[key][k].append(float(v))
with torch.no_grad():
    for i in range(N):
        r=valid.iloc[i]
        sL=load_latent(r["starting_image_path"]).unsqueeze(0).to(DEV); fL=load_latent(r["followup_image_path"]).unsqueeze(0).to(DEV)
        ctx=ctx_of(r); dt=(float(r["followup_follow_up"])-float(r["starting_follow_up"]))/12.0
        base=dec(sL).cpu().numpy(); real=dec(fL).cpu().numpy()
        sz=sL*sf; age=torch.tensor([dt]).view(1,1,1,1,1).expand(1,1,*sz.shape[-3:]).to(DEV); cond=torch.cat([sz,age],dim=1)
        z=sched.add_noise(sz,torch.randn_like(sz),torch.tensor([T_START]).to(DEV))
        for t in [x for x in sched.timesteps if x<=T_START]:
            tsb=torch.tensor([t]).to(DEV)
            dh,mh=cnet(x=z.float(),timesteps=tsb,context=ctx,controlnet_cond=cond.float())
            eps=diff(x=z.float(),timesteps=tsb,context=ctx.float(),down_block_additional_residuals=dh,mid_block_additional_residual=mh)
            z,_=sched.step(eps,t,z)
        gen=dec(z/sf).cpu().numpy()
        add("method", gen, real, base)
        add("copy",  base, real, base)                                   # predict no change
        rng=np.random.RandomState(i); add("noise", np.clip(base+rng.randn(*base.shape)*0.03,0,1), real, base)
        print(f"[{i+1}/{N}]",flush=True)
def rep(k):
    a=acc[k]; f=lambda x: (np.mean(x) if x else float('nan'))
    return f"CHANGE_PCC {f(a['pcc']):.3f}  CHANGE_DICE {f(a['dice']):.3f}  CHANGE_MAE {f(a['mae']):.4f}  (n={len(a['pcc'])})"
print(f"\n===== FIELD-STANDARD CHANGE METRICS (detrended, direction-aware, N={N}) =====")
print(f"SDEdit-MambaControl : {rep('method')}")
print(f"COPY (no-change)    : {rep('copy')}")
print(f"NOISE control (3%)  : {rep('noise')}")
print("PCC>0 & DICE>copy & MAE<noise => method captures real progression direction.")
print("CHANGEMETRIC_DONE")
