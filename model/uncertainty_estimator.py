import os
import torch
import einops
from torch import nn
from typing import Tuple 
from termcolor import colored

from utils import freeze_model

class LinearEstimator(nn.Module):
    def __init__(self,
                 name:str,
                 in_dim: int,
                 conditioned: bool = False,
                 train_encoder: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None) -> None:
        super(LinearEstimator, self).__init__()
        self.encoder = train_encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conditioned = conditioned
        if self.conditioned:
            in_dim += 384 # text_dim
            print(colored("[LinearEstimator] ", "yellow") + f"Uncertainty estimator is conditioned on text")

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.trainable_params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optimizer(params=self.trainable_params)

    def forward(self, feat: torch.Tensor, text_cond: str = None) -> torch.Tensor:
        if self.encoder is None:
            feat = einops.rearrange(feat, 'b n c -> b c n')
            feat = self.pool(feat)
            feat = feat.squeeze(-1)  
        if self.conditioned:
            assert text_cond is not None, "Text condition must be provided for conditioned estimator." 
            feat = torch.cat([feat, text_cond], dim=-1)
        return self.fc(feat).squeeze(-1) 

    def get_loss(
            self,
            observations: torch.Tensor,
            uncertainties: torch.Tensor,
            text_cond: torch.Tensor = None,
            encoder: nn.Module = None
            ) -> Tuple[torch.Tensor, dict]:

        with torch.autocast(device_type='cuda'):
            if self.encoder is None: # Using pretrained
                with torch.no_grad():
                    latent_dict = encoder.forward_features(observations)
                feat = latent_dict['x_norm_patchtokens']
            else:
                feat = self.encoder(observations)
            preds = self.forward(feat, text_cond=text_cond)
            loss = nn.functional.mse_loss(preds.squeeze(), uncertainties.squeeze())
            aux_losses = {'total_loss': loss}

        return loss, aux_losses
    
    def step_optimizer(self, device: torch.device, 
                       observations: torch.Tensor, 
                       uncertainties: torch.Tensor,
                       text_cond: torch.Tensor = None,
                       encoder: nn.Module = None,
                       scaler: torch.cuda.amp.GradScaler = None) -> dict:

        self.optimizer.zero_grad()
        loss, aux_losses = self.get_loss(observations, uncertainties, text_cond=text_cond, encoder=encoder)

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

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




