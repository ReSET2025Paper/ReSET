"""
Code partially taken from spatial temporal transformer from
@article{gao2025adaworld,
  title={AdaWorld: Learning Adaptable World Models with Latent Actions}, 
  author={Gao, Shenyuan and Zhou, Siyuan and Du, Yilun and Zhang, Jun and Gan, Chuang},
  journal={arXiv preprint arXiv:2503.18938},
  year={2025}
}
"""
import os
import torch
import einops
import torch.nn.functional as F

from torch import nn
from typing import Tuple
from termcolor import colored
from utils import freeze_model
from . import SpatioTemporalTransformer, patchify, unpatchify


class FlowDecoder(nn.Module):
    def __init__(self, 
                 name: str,
                 in_dim: int,
                 out_dim: int,
                 model_dim: int,
                 num_blocks: int, 
                 num_heads: int,
                 patch_size: int = 5,
                 pred_timestep: int = 6,
                 conditioned: bool = False,
                 train_encoder: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 schedulers: Tuple[torch.optim.lr_scheduler.LinearLR, 
                                   torch.optim.lr_scheduler.CosineAnnealingLR] = None,
                 milestones: list = [10],
                 ) -> None:
        super(FlowDecoder, self).__init__()

        self.encoder = train_encoder
        self.conditioned = conditioned

        self.patch_size = patch_size
        self.pred_timestep = pred_timestep
        # self.time_embedding = nn.Parameter(torch.randn(1, pred_timestep, 1, 1, 1))  # [1, T, 1, 1, 1]

        self.decoder = SpatioTemporalTransformer(
            in_dim = in_dim * patch_size**2,
            out_dim = out_dim * patch_size**2, 
            model_dim=model_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            
        )

        if self.conditioned:
            print(colored("[FlowDecoder] ", "yellow") + f"Flow predictor is conditioned on text")
            self.cond_proj = nn.Linear(384, in_dim * patch_size**2)

        # get trainable parameters
        self.trainable_params = [p for p in self.parameters() if p.requires_grad]

        self.optimizer = optimizer(params=self.trainable_params)
        schedulers = [scheduler(optimizer=self.optimizer) for scheduler in schedulers]
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                                                                schedulers=schedulers,
                                                                milestones=milestones
                                                                ) 

    def forward(self, patch_tokens: torch.Tensor, text_cond: torch.Tensor = None) -> torch.Tensor:
        spatial_feat = einops.rearrange(
            patch_tokens, 'b (h w) c -> b c h w', h=int(patch_tokens.shape[1]**0.5), w=int(patch_tokens.shape[1]**0.5))
        spatial_feat = nn.functional.interpolate(spatial_feat, size=(20, 20), mode='bilinear')
        spatial_feat = einops.repeat(spatial_feat.unsqueeze(1), 'b 1 c h w -> b t c h w', t=self.pred_timestep)
        spatial_feat = einops.rearrange(spatial_feat, 'b t c h w -> b t h w c')
        # spatial_feat = spatial_feat + self.time_embedding

        B, T, H, W = spatial_feat.shape[:4]
        patches = patchify(spatial_feat, self.patch_size)

        if self.conditioned:
            assert text_cond is not None, "Text condition must be provided for conditioned decoder."
            cond = self.cond_proj(text_cond)  # shape: [B, in_dim * patch_size^2]
            cond = cond.unsqueeze(1).unsqueeze(2)  # shape: [B, 1, 1, C]
            cond = cond.expand(-1, patches.shape[1], patches.shape[2], -1)  # broadcast to [B, T, N, C]
            patches = patches + cond

        flow = self.decoder(patches)
        # flow = F.sigmoid(flow)
        # flow = torch.clamp(flow, 0.0, 1.0)
        # flow = F.tanh(flow)
        flow = unpatchify(flow, self.patch_size, H, W)
        flow = einops.rearrange(flow, 'b t h w c -> b c t h w')

        return flow
    
    def get_loss(
            self, 
            init_obs: torch.Tensor,
            tracking: torch.Tensor,
            text_cond: torch.Tensor = None,
            encoder: nn.Module = None
    ) -> Tuple[torch.Tensor, dict]:
        
        with torch.autocast(device_type='cuda'):
            if self.encoder is None:  # Using pretrained
                with torch.no_grad():
                    feat = encoder.forward_features(init_obs)['x_norm_patchtokens']
            else: 
                feat = self.encoder(init_obs)
            pred_flow = self.forward(feat, text_cond=text_cond)
            loss = nn.functional.mse_loss(pred_flow, tracking)
            # loss = nn.functional.huber_loss(pred_flow, tracking)
            aux_losses = {
                'total_loss': loss,
                'mse_loss': loss,
            }
            return loss, aux_losses, pred_flow
    
 
    def step_optimizer(self,
                       device: str, 
                       init_obs: torch.Tensor,
                       tracking: torch.Tensor,
                       text_cond: torch.Tensor = None,
                       encoder: nn.Module = None,
                       scaler: torch.amp.GradScaler = None,
                       clip_val: float = None,
                       ) -> dict:
        
        self.optimizer.zero_grad()
        loss, aux_losses, _ = self.get_loss(init_obs, tracking, text_cond=text_cond, encoder=encoder)

        scaler.scale(loss).backward()

        if clip_val is not None:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)

        scaler.step(self.optimizer)
        scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
            aux_losses['learning_rate'] = torch.tensor(self.scheduler.get_last_lr())

        return aux_losses
    
    def log_ckpt(self, epoch_num: int, type: str) -> None:
        ckpt_path = "ckpt"
        os.makedirs(ckpt_path, exist_ok=True)

        if type == "last":
            torch.save(self.state_dict(), os.path.join(ckpt_path, "model_last.pth"))
        elif type == "log":
            torch.save(self.state_dict(), os.path.join(ckpt_path, f"model_step{epoch_num:05}.ckpt"))
        elif type == "best":
            torch.save(self.state_dict(), os.path.join(ckpt_path, "model_best.pth"))

