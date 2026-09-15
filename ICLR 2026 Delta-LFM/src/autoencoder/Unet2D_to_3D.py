import os
from typing import Optional
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

# ---------- Utility: Load checkpoint ----------
def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    if checkpoints_path is not None:
        if not os.path.exists(checkpoints_path):
            print(f"{checkpoints_path} does not exist, using default weights.")

        checkpoint = torch.load(checkpoints_path, map_location='cpu')
        new_state_dict = {}

        if "autoencoder" in checkpoints_path:
            for key in checkpoint:
                new_key = key.replace("conv.conv", "postconv.conv") if "conv.conv" in key else key
                new_state_dict[new_key] = checkpoint[key]

        elif "controlnet" in checkpoints_path or "cnet" in checkpoints_path:
            for key, val in checkpoint.items():
                key = key.replace("module.", "")
                new_key = key.replace("to_out.0", "out_proj") if "to_out.0" in key else key
                new_state_dict[new_key] = val

        elif "diffusion" in checkpoints_path:
            for k, v in checkpoint.items():
                new_k = k.replace("to_out.0", "out_proj") \
                         .replace("proj_out.0", "proj_out.conv") \
                         .replace("proj_in.0", "proj_in.conv") \
                         .replace("conv_in.0", "conv_in.conv") \
                         .replace("conv_out.0", "out.2.conv") \
                         .replace("time_embedding.linear_1", "time_embed.0") \
                         .replace("time_embedding.linear_2", "time_embed.2")
                new_state_dict[new_k] = v

        print("Loaded pretrained weights from", checkpoints_path)
        network.load_state_dict(new_state_dict)
    return network

# ---------- Residual Block ----------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.same_channels = in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if not self.same_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.skip_conv(x) + self.block(x)

# ---------- VAE-style 3D Latent Fuser ----------
class LatentFuser3D(nn.Module):
    def __init__(self, in_channels=8, latent_channels=8):
        super().__init__()
        self.hidden_depth = [128, 64, 32, 16]

        # Initial encoder
        self.encoder3d = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_depth[0], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(self.hidden_depth[0], self.hidden_depth[0], kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Residual transform
        self.transform = nn.Sequential(*[
            ResidualBlock3D(self.hidden_depth[i], self.hidden_depth[i + 1])
            for i in range(len(self.hidden_depth) - 1)
        ])

        # Latent projection
        self.fc_mu = nn.Conv3d(self.hidden_depth[-1], latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv3d(self.hidden_depth[-1], latent_channels, kernel_size=1)

        # Reverse transformation
        self.recover_transform = nn.Sequential(
            nn.Conv3d(latent_channels, self.hidden_depth[-1], kernel_size=1),
            nn.GELU(),
            *[
                ResidualBlock3D(self.hidden_depth[i], self.hidden_depth[i - 1])
                for i in range(len(self.hidden_depth) - 1, 0, -1)
            ]
        )

        # Final decoder
        self.decoder3d = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_depth[0], in_channels, kernel_size=1),
            # nn.GELU(),
        )

    def encode(self, x):
        x = self.encoder3d(x)
        x = self.transform(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z_log_var = torch.clamp(logvar, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return mu, z_sigma

    def reparameterize(self, mu, std):
        # std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.recover_transform(z)
        return self.decoder3d(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu # self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------- Autoencoder wrapper ----------
class Autoencoder2Dto3D(nn.Module):
    def __init__(self, autoencoder_2d: nn.Module, fusion_adapter_3d: nn.Module):
        super().__init__()
        self.autoencoder_2d = autoencoder_2d
        self.fusion_adapter_3d = fusion_adapter_3d

    def encode_slices(self, x_3d: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x_3d.shape
        latents = []
        for d in range(D):
            slice_2d = x_3d[:, :, d].repeat(1, 3, 1, 1)  # [B, 3, H, W]
            latent_dist = self.autoencoder_2d.encode(slice_2d)
            latent = latent_dist.latent_dist.mean  # [B, 4, H//8, W//8]
            latents.append(latent)
        return torch.stack(latents, dim=2)  # [B, C_latent, D, H', W']

    def decode_slices(self, latent_3d: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = latent_3d.shape
        recons = []
        for d in range(C):  # D is 3
            latent_2d = latent_3d[:, d]
            recon = self.autoencoder_2d.decode(latent_2d).sample
            recon = recon.mean(dim=1, keepdim=True)  # [B, 1, H*8, W*8]
            recons.append(recon)
        return torch.stack(recons, dim=2)  # [B, 1, D, H*8, W*8]

    def forward(self, latent):
        recon_latent, mu, logvar = self.fusion_adapter_3d(latent)
        return recon_latent, mu, logvar

    def encode(self, x_3d: torch.Tensor) -> torch.Tensor:
        latent = self.encode_slices(x_3d)
        mu, logvar = self.fusion_adapter_3d.encode(latent)
        z = self.fusion_adapter_3d.reparameterize(mu, logvar)
        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = self.fusion_adapter_3d.decode(latent)
        return self.decode_slices(latent)



# ---------- Initialization ----------
def init_autoencoder(args: Optional[str] = None, slice_number=128) -> nn.Module:
    vae_model = "stabilityai/sd-vae-ft-ema"
    autoencoder_2d = AutoencoderKL.from_pretrained(vae_model, force_upcast=True).eval()
    for param in autoencoder_2d.parameters():
        param.requires_grad = False

    latent_fuser_3d = LatentFuser3D(in_channels=slice_number, latent_channels=8)
    latent_fuser_3d = load_if(args.aekl_ckpt, latent_fuser_3d)

    return Autoencoder2Dto3D(autoencoder_2d=autoencoder_2d, fusion_adapter_3d=latent_fuser_3d)



# ---------- Test ----------
if __name__ == "__main__":
    import time
    slice_number = 128
    autoencoder = init_autoencoder(slice_number=slice_number).eval()
    img = torch.randn(1, 1, slice_number, 144, 128)

    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        img = img.cuda()

    start_time = time.time()
    with torch.no_grad():

        latent = autoencoder.encode(img)
        print("latent shape:", latent.shape)
        recon = autoencoder.decode(latent)
        print("recon shape:", recon.shape)

    print("time elapsed:", time.time() - start_time)

