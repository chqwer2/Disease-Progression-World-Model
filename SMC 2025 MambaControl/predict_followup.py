"""MambaControl disease-progression inference (single-model SDEdit + conservative blend).

Given a BASELINE scan latent + target time (+ baseline covariates, all available at inference),
predict the follow-up scan. Pipeline:
  1. SDEdit: noise the baseline latent to t_start, denoise with diffusion + v3 controlnet
     conditioned on [baseline_latent, delta_t] (spatial) and covariates (cross-attn).
  2. Conservative blend: followup = baseline + ALPHA * (sdedit_prediction - baseline).
     ALPHA<1 scales down a prediction whose direction is more reliable than its magnitude,
     so the blend stays close to the copy baseline instead of over-shooting.

No target/follow-up information is used -> valid at deployment.
Usage: python predict_followup.py --row <i>            # eval a held-out pair, prints metrics
       python predict_followup.py --start_latent a.npz --delta_t_years 2.0 --out pred.npy
OPERATING_POINT: t_start / steps / alpha are set from the sweep (eval_final.py).
"""
import argparse, os, numpy as np, torch, pandas as pd, torch.nn.functional as F
from monai.networks.schedulers import DDIMScheduler
from mambacontrol import networks
from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder
DEV="cuda"
# Checkpoints — override with MC_AE_CKPT / MC_UNET_CKPT / MC_CNET_CKPT / MC_PAIRS_CSV.
AE  =os.environ.get("MC_AE_CKPT",  "checkpoints/autoencoder.pth")
UNET=os.environ.get("MC_UNET_CKPT","checkpoints/unet.pth")
CNET=os.environ.get("MC_CNET_CKPT","checkpoints/controlnet.pth")
# ---- OPERATING POINT: pick these with the eval_final.py sweep, then set them here. ----
# Halving STEPS roughly halves inference cost; re-run the sweep before changing it.
T_START = 200
STEPS   = 100
ALPHA   = 0.10
PAIRS = os.environ.get("MC_PAIRS_CSV", "data/pairs.csv")

def build():
    ae=init_autoencoder(AE).to(DEV).float().eval()
    for m in ae.modules():
        if hasattr(m,"norm_float16"): m.norm_float16=False
    diff=networks.init_mamba_diffusion(UNET).to(DEV).eval()
    cnet=networks.init_mamba_controlnet(); cnet.load_state_dict(torch.load(CNET,map_location="cpu")); cnet=cnet.to(DEV).eval()
    return ae,diff,cnet

def load_latent(path):
    z=torch.from_numpy(np.load(path.replace(".nii.gz","_new_latent.npz").replace(".nii","_new_latent.npz"))["data"]).float()
    pads=[]
    for d in reversed(z.shape[1:]): pads+=[0,(-d)%4]
    return F.pad(z,pads)

@torch.no_grad()
def predict(ae,diff,cnet, start_latent, delta_t_years, context, sf):
    """start_latent: unscaled (C,D,H,W) tensor. Returns predicted followup IMAGE (D,H,W) in [0,1]."""
    sched=DDIMScheduler(num_train_timesteps=1000,schedule="scaled_linear_beta",beta_start=0.0015,beta_end=0.0205)
    sched.set_timesteps(STEPS)
    sL=start_latent.unsqueeze(0).to(DEV); sz=sL*sf
    age=torch.tensor([delta_t_years]).view(1,1,1,1,1).expand(1,1,*sz.shape[-3:]).to(DEV)
    cond=torch.cat([sz,age],dim=1); ctx=context.to(DEV)
    z=sched.add_noise(sz,torch.randn_like(sz),torch.tensor([T_START]).to(DEV))
    for t in [x for x in sched.timesteps if x<=T_START]:
        tsb=torch.tensor([t]).to(DEV)
        dh,mh=cnet(x=z.float(),timesteps=tsb,context=ctx,controlnet_cond=cond.float())
        eps=diff(x=z.float(),timesteps=tsb,context=ctx.float(),down_block_additional_residuals=dh,mid_block_additional_residual=mh)
        z,_=sched.step(eps,t,z)
    base=ae.decode_stage_2_outputs(sL.float())[0,0].clamp(0,1)
    gen =ae.decode_stage_2_outputs((z/sf).float())[0,0].clamp(0,1)
    followup=(base + ALPHA*(gen-base)).clamp(0,1)   # conservative blend
    return followup, base

def ctx_of(r):
    return torch.tensor([[[float(r['followup_age']),0.5,float(r['starting_diagnosis']),float(r['starting_cerebral_cortex']),float(r['starting_hippocampus']),float(r['starting_amygdala']),float(r['starting_cerebral_white_matter']),float(r['starting_lateral_ventricle'])]]],dtype=torch.float32)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--row",type=int,default=None,help="held-out pair index -> print metrics vs copy")
    ap.add_argument("--start_latent",type=str,default=None); ap.add_argument("--delta_t_years",type=float,default=None)
    ap.add_argument("--out",type=str,default=None)
    a=ap.parse_args()
    ae,diff,cnet=build()
    valid=pd.read_csv(PAIRS); valid=valid.iloc[int(0.8*len(valid)):].reset_index(drop=True)
    sf=float(1.0/torch.std(load_latent(valid["followup_image_path"].iloc[0])))
    print(f"Operating point: t_start={T_START} steps={STEPS} alpha={ALPHA}")
    if a.row is not None:
        from monai.metrics import PSNRMetric, SSIMMetric
        psnr=PSNRMetric(max_val=1.0); ssim=SSIMMetric(spatial_dims=3,data_range=1.0,win_size=7)
        r=valid.iloc[a.row]; sL=load_latent(r["starting_image_path"]); fL=load_latent(r["followup_image_path"])
        dt=(float(r["followup_follow_up"])-float(r["starting_follow_up"]))/12.0
        pred,base=predict(ae,diff,cnet,sL,dt,ctx_of(r),sf)
        real=ae.decode_stage_2_outputs(fL.unsqueeze(0).float().to(DEV))[0,0].clamp(0,1)
        Pp=lambda x:float(psnr(x[None,None],real[None,None]).item()); Ss=lambda x:float(ssim(x[None,None],real[None,None]).item())
        print(f"row {a.row}: pred PSNR {Pp(pred):.3f} SSIM {Ss(pred):.4f} | copy PSNR {Pp(base):.3f} SSIM {Ss(base):.4f}")
        if a.out: np.save(a.out, pred.cpu().numpy()); print("saved",a.out)
    elif a.start_latent and a.delta_t_years is not None:
        sL=load_latent(a.start_latent)
        ctx=torch.zeros(1,1,8); ctx[0,0,1]=0.5  # sex only known; other covariates 0 if unavailable
        pred,_=predict(ae,diff,cnet,sL,a.delta_t_years,ctx,sf)
        out=a.out or "followup_pred.npy"; np.save(out, pred.cpu().numpy()); print("saved",out)
    else:
        print("provide --row I  OR  --start_latent x.npz --delta_t_years T [--out f.npy]")
