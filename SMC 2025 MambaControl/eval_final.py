"""Operating-point eval for the single-model SDEdit variant of MambaControl.
Sweeps (a) the test-time config -- t_start x sampling-steps -- and (b) alpha (prediction
strength), and reports PSNR/SSIM/corr against the copy baseline. Checkpoint via argv;
sweep ranges via env (T_STARTS, STEPS_LIST, ALPHAS, N; comma-separated). Run it per
ControlNet checkpoint to compare training lengths."""
import os, sys, numpy as np, torch, pandas as pd, torch.nn.functional as F
from monai.metrics import PSNRMetric, SSIMMetric
from monai.networks.schedulers import DDIMScheduler
from mambacontrol import networks
from mambacontrol.autoencoder.MAISI_Unet3D import init_autoencoder
DEV="cuda"
AE  =os.environ.get("MC_AE_CKPT",  "checkpoints/autoencoder.pth")
UNET=os.environ.get("MC_UNET_CKPT","checkpoints/unet.pth")
CNET=sys.argv[1]
T_STARTS=[int(x) for x in os.environ.get("T_STARTS","150,200,250").split(",")]
STEPS_LIST=[int(x) for x in os.environ.get("STEPS_LIST","25,50,100").split(",")]
ALPHAS=[float(x) for x in os.environ.get("ALPHAS","0.05,0.10,0.15").split(",")]
N=int(os.environ.get("N","20"))
valid=pd.read_csv(os.environ.get("MC_PAIRS_CSV","data/pairs.csv"))
valid=valid.iloc[int(0.8*len(valid)):].reset_index(drop=True)
ae=init_autoencoder(AE).to(DEV).float().eval()
for m in ae.modules():
    if hasattr(m,"norm_float16"): m.norm_float16=False
diff=networks.init_mamba_diffusion(UNET).to(DEV).eval()
cnet=networks.init_mamba_controlnet(); cnet.load_state_dict(torch.load(CNET,map_location="cpu")); cnet=cnet.to(DEV).eval()
def load_latent(path):
    z=torch.from_numpy(np.load(path.replace(".nii.gz","_new_latent.npz").replace(".nii","_new_latent.npz"))["data"]).float()
    pads=[]
    for d in reversed(z.shape[1:]): pads+=[0,(-d)%4]
    return F.pad(z,pads)
sf=float(1.0/torch.std(load_latent(valid["followup_image_path"].iloc[0])))
psnr=PSNRMetric(max_val=1.0); ssim=SSIMMetric(spatial_dims=3,data_range=1.0,win_size=7)
def dec(latent): return ae.decode_stage_2_outputs(latent.float())[0,0].clamp(0,1)
def P(x,y): return float(psnr(x[None,None],y[None,None]).item())
def S(x,y): return float(ssim(x[None,None],y[None,None]).item())
def ctx_of(r): return torch.tensor([[[float(r['followup_age']),0.5,float(r['starting_diagnosis']),float(r['starting_cerebral_cortex']),float(r['starting_hippocampus']),float(r['starting_amygdala']),float(r['starting_cerebral_white_matter']),float(r['starting_lateral_ventricle'])]]],dtype=torch.float32).to(DEV)

res={(T,K,a):{"ps":[],"ss":[]} for T in T_STARTS for K in STEPS_LIST for a in ALPHAS}
corr={(T,K):[] for T in T_STARTS for K in STEPS_LIST}; copy_ps,copy_ss=[],[]
scheds={K:DDIMScheduler(num_train_timesteps=1000,schedule="scaled_linear_beta",beta_start=0.0015,beta_end=0.0205) for K in STEPS_LIST}
for K in STEPS_LIST: scheds[K].set_timesteps(K)
with torch.no_grad():
    for i in range(N):
        r=valid.iloc[i]
        sL=load_latent(r["starting_image_path"]).unsqueeze(0).to(DEV); fL=load_latent(r["followup_image_path"]).unsqueeze(0).to(DEV)
        ctx=ctx_of(r); dt=(float(r["followup_follow_up"])-float(r["starting_follow_up"]))/12.0
        base=dec(sL); real=dec(fL); copy_ps.append(P(base,real)); copy_ss.append(S(base,real))
        sz=sL*sf; age=torch.tensor([dt]).view(1,1,1,1,1).expand(1,1,*sz.shape[-3:]).to(DEV); cond=torch.cat([sz,age],dim=1)
        noise=torch.randn_like(sz)
        for K in STEPS_LIST:
            sch=scheds[K]
            for T in T_STARTS:
                z=sch.add_noise(sz,noise,torch.tensor([T]).to(DEV))
                for t in [x for x in sch.timesteps if x<=T]:
                    tsb=torch.tensor([t]).to(DEV)
                    dh,mh=cnet(x=z.float(),timesteps=tsb,context=ctx,controlnet_cond=cond.float())
                    eps=diff(x=z.float(),timesteps=tsb,context=ctx.float(),down_block_additional_residuals=dh,mid_block_additional_residual=mh)
                    z,_=sch.step(eps,t,z)
                gen=dec(z/sf); p=gen-base
                pc=p.flatten().cpu().numpy(); rcf=(real-base).flatten().cpu().numpy()
                if rcf.std()>1e-6 and pc.std()>1e-6: corr[(T,K)].append(float(np.corrcoef(rcf,pc)[0,1]))
                for a in ALPHAS:
                    bl=(base+a*p).clamp(0,1); res[(T,K,a)]["ps"].append(P(bl,real)); res[(T,K,a)]["ss"].append(S(bl,real))
        print(f"[{i+1}/{N}]",flush=True)
cp=np.mean(copy_ps); cs=np.mean(copy_ss)
print(f"\n### CKPT {CNET.split('/')[-1]} | COPY PSNR {cp:.3f} SSIM {cs:.4f} (N={N})")
print(f"{'t_start':>7}{'steps':>6}{'alpha':>6}{'PSNR':>9}{'dPSNR':>8}{'SSIM':>9}{'dSSIM':>9}{'corr':>7}")
best=None
for T in T_STARTS:
    for K in STEPS_LIST:
        c=np.mean(corr[(T,K)]) if corr[(T,K)] else float('nan')
        for a in ALPHAS:
            p=np.mean(res[(T,K,a)]["ps"]); s=np.mean(res[(T,K,a)]["ss"])
            flag="  <<>copy" if (p>cp and s>cs) else ""
            print(f"{T:>7}{K:>6}{a:>6.2f}{p:>9.3f}{p-cp:>+8.3f}{s:>9.4f}{s-cs:>+9.4f}{c:>7.3f}{flag}")
            if p>cp and s>cs and (best is None or (p-cp)+(s-cs)*20>best[0]): best=((p-cp)+(s-cs)*20,T,K,a,p,s,c)
if best: print(f">>> BEST beats-copy: t_start={best[1]} steps={best[2]} alpha={best[3]:.2f} -> PSNR {best[4]:.3f}(+{best[4]-cp:.3f}) SSIM {best[5]:.4f}(+{best[5]-cs:.4f}) corr {best[6]:.3f}")
else: print(">>> none beat copy on both")
print("FINAL_EVAL_DONE")
