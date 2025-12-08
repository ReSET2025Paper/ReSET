# The dynamics model is used for the Dynamics-DP baseline.
# It predicts the next state given the current state and action.

"""
Dynamics model adapted from the latent action model from: 
@article{gao2025adaworld,
  title={AdaWorld: Learning Adaptable World Models with Latent Actions}, 
  author={Gao, Shenyuan and Zhou, Siyuan and Du, Yilun and Zhang, Jun and Gan, Chuang},
  journal={arXiv preprint arXiv:2503.18938},
  year={2025}
}
The baseline implemented is based on the paper:
@inproceedings{wu2025neural,
    title={Neural Dynamics Augmented Diffusion Policy},
    author={Wu, Ruihai and Chen, Haozhe and Zhang, Mingtong and Lu, Haoran and Li, Yitong and Li, Yunzhu},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    year={2025}
}
"""

import os
import cv2
import piq
import torch
import einops
import numpy as np
import torch.nn.functional as F

from torch import nn
from PIL import Image
from typing import Dict, Tuple, Union
from termcolor import colored
from . import SpatioTransformer, patchify, unpatchify

class DynamicsModel(nn.Module):
    def __init__(self,         
                name: str,   
                device: torch.device,
                in_dim: int,
                model_dim: int,
                patch_size: int,
                dec_blocks: int,
                num_heads: int,
                beta: float = 0.01,
                optimizer: torch.optim = torch.optim.AdamW,
                dropout: float = 0.0,) -> None:
        super(DynamicsModel, self).__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        # self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        # nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.action_proj = nn.Linear(10, model_dim)  # 8 is the action dimension (3 start state + 3 end state + 1 rotation + 3 type logits)
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        # self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        # Added code
        self.optimizer = optimizer(self.parameters()) # init_args set in config
        self.beta = beta
        self.device = device

    def forward(self, obs: dict, preset: Union[dict, torch.Tensor]) -> Dict:
        init_obs = torch.tensor(obs["img"]).to(self.device) / 255
        if len(init_obs.shape) == 4:  
            init_obs = einops.repeat(init_obs, 'b h w c -> b t h w c', t=1)
        if init_obs.shape[-1] != 3:
            init_obs = einops.rearrange(init_obs, 'b t c h w -> b t h w c')
        assert len(init_obs.shape) == 5 and init_obs.shape[-1] == 3, "Input videos should be of shape (B, T, H, W, C)"
        H, W = init_obs.shape[2], init_obs.shape[3]

        if type(preset) is dict:
            action_type = preset['type']
            action = np.concatenate((
                np.array(preset['start_state']),
                np.array(preset['end_state']),
                np.expand_dims(np.array(preset['rotation']), axis=1),
                action_type
            ), axis=1)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action_tensor = preset.to(self.device)
        
        video_patches = patchify(init_obs, self.patch_size)
        video_patches = self.patch_up(video_patches) 
        action_feat = self.action_proj(action_tensor)  # (B, E)
        action_feat = einops.repeat(action_feat, 'b e -> b t 1 e', t=video_patches.shape[1])

        video_action_patches = video_patches + action_feat

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = F.sigmoid(video_recon)
        video_recon = unpatchify(video_recon, self.patch_size, H, W)

        return video_recon

    def get_loss(self, batch: Dict) -> Tuple[Dict, torch.Tensor, Tuple]: # same type as outputs
        video_recon = self(batch['obs'], batch['action'])
        goal_obs = torch.tensor(batch["obs"]["goal_img"]).to(self.device) / 255
        if goal_obs.shape[-1] != 3:
            goal_obs = einops.rearrange(goal_obs, 'b c h w -> b h w c')
        if len(goal_obs.shape) == 4:
            goal_obs = einops.repeat(goal_obs, 'b h w c -> b t h w c', t=video_recon.shape[1])
        assert len(goal_obs.shape) == 5 and goal_obs.shape[-1] == 3, "Input videos should be of shape (B, H, W, C)"

        # Compute loss
        mse_loss = ((goal_obs - video_recon) ** 2).mean()
        loss = mse_loss

        # Compute monitoring measurements
        gt = goal_obs.clamp(0, 1).reshape(-1, *goal_obs.shape[2:]).permute(0, 3, 1, 2)
        recon = video_recon.clamp(0, 1).reshape(-1, *video_recon.shape[2:]).permute(0, 3, 1, 2)
        psnr = piq.psnr(gt, recon).mean()
        ssim = piq.ssim(gt, recon).mean()
        return loss, {
            "mse_loss": mse_loss,
            "total_loss": loss,
            "psnr": psnr,
            "ssim": ssim
        }, video_recon
    
    def step_optimizer(self, 
                       batch: Dict,
                       scaler: torch.amp.GradScaler = None,
                       clip_val=None) -> None:

        self.optimizer.zero_grad()
        loss, aux_losses, _ = self.get_loss(batch)
        with torch.amp.autocast("cuda"):
            scaler.scale(loss).backward()
        
        if clip_val is not None:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
        
        scaler.step(self.optimizer)
        scaler.update()
        return aux_losses

    def log_ckpt(self, epoch_num: int, model_type: str) -> None:
        ckpt_path = "ckpt"
        os.makedirs(ckpt_path, exist_ok=True)

        if model_type == "last":
            torch.save(self.state_dict(), os.path.join(ckpt_path, "model_last.pth"))
        elif model_type == "log":
            torch.save(self.state_dict(), os.path.join(ckpt_path, f"model_step{epoch_num:05}.ckpt"))
        elif model_type == "best":
            torch.save(self.state_dict(), os.path.join(ckpt_path, "model_best.pth"))
    
    def log_images(self, batch: Dict, outputs: Dict, split: str, epoch_num: int) -> None:
        gt_seq = batch["obs"]["img"][0].cpu().unsqueeze(0)
        gt_seq = torch.cat([gt_seq, batch["obs"]["goal_img"][0].cpu().unsqueeze(0)], dim=0)
        gt_seq = einops.rearrange(gt_seq, 't c h w -> t h w c')
        recon_seq = outputs[0].cpu()
        recon_seq = torch.cat([gt_seq[:1], recon_seq * 255], dim=0)
        compare_seq = torch.cat([gt_seq, recon_seq], dim=1)
        compare_seq = einops.rearrange(compare_seq, "t h w c -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        compare_seq = cv2.cvtColor(compare_seq, cv2.COLOR_BGR2RGB)
        img_path = os.path.join(f"./logged_images/{split}_step{epoch_num:06}.png")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        try:
            img.save(img_path)
        except:
            pass

        
