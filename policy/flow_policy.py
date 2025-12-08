import os
import torch
import einops
import numpy as np
from torch import nn
from termcolor import colored
from policy import CorrectionPolicy

import cv2
from utils import visualize_pred_tracking

class FlowPolicy(CorrectionPolicy):
    def __init__(self,
                 name: str = "FlowPolicy",
                 encoder: torch.nn.Module = None,
                 flow_steps: int = 18,
                 flow_shape: tuple = (20, 20),
                 token_dim: int = 128,
                 optimizer: torch.optim.Optimizer = None,
                 **kwargs) -> None:
        super(FlowPolicy, self).__init__(**kwargs)
        self.flow_proj = nn.Sequential(
            nn.Linear(2 * (flow_steps - 1) * flow_shape[0] // 2 * flow_shape[1] // 2, 256), # -1 for relative flow
            nn.ReLU(),
            nn.Linear(256, token_dim)
        )
        
        # Image encoder
        assert encoder is not None and optimizer is not None, "Encoder and optimizer must be provided for parameterized preset policy."
        self.encoder = encoder
        dummy_input = torch.randn(1, 3, 224, 224)
        if any(p.requires_grad for p in self.encoder.parameters()): # Using resnet
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final fc layer
            with torch.no_grad():
                encoder_out_dim = self.encoder(dummy_input).squeeze().shape[0]
        self.img_proj = nn.Sequential(
                 nn.Linear(encoder_out_dim, 512),
                 nn.ReLU(),
                 nn.Linear(512, 256),
                 nn.ReLU(),
                 nn.Linear(256, token_dim))

        self.action_token = nn.Parameter(torch.randn(1, 1, token_dim), requires_grad=True)  # Action token for the transformer
  
        # Transformer policy
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output head
        self.action_projector = nn.Sequential(
                 nn.Linear(token_dim, 64),
                 nn.ReLU(),
                 nn.Linear(64, 7)  # Start state (3) + end state (3) + rotation (1)
            )
        self.type_projector = nn.Linear(token_dim, 3) # 3 logits: pull, pick, rotate

        self.trainable_params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optimizer(params=self.trainable_params)
        print(colored("[FlowPolicy] Initialized with trainable parameters:", "green"), len(self.trainable_params))

        self.action_map = {
            'pull': self._pull,
            'pick': self._pick,
            'rotate': self._rotate
        }
        self.types = ['pull', 'pick', 'rotate']

    @torch.no_grad()
    def get_action(self, obs_dict: dict[str, np.array]) -> tuple[np.ndarray, bool]:
        assert self.robot is not None, "Robot must be provided for FlowPolicy" # NOTE: Because of positional control
        img = obs_dict['image'][-1]
        text = input(colored("[FlowPolicy] ", "yellow") + "Enter the task to perform: ") if self.conditioned else None
        flow, uncertainty = self._get_flow_and_uncertainty(img, text)
        print(colored("[PresetPolicy] ", "green") + f"Current uncertainty is: {uncertainty}")
        obs = {
            'flow': flow,
            'img': img
        }
        if uncertainty <= self.uncertainty_threshold:
            return None, True # correction finished
        else:
            obs_flow = visualize_pred_tracking(
                        img / 255,
                        flow,
                        relative_flow=True,
                        denoise_flow=True,
            )
            print(len(obs_flow))
            for idx in range(len(obs_flow)):
                cv2.imshow("flow", obs_flow[idx])
                cv2.waitKey(100)
            cv2.waitKey(0)
            interaction_spec = self.predict_action(obs)
            interaction_spec['type'] = self.types[interaction_spec['type']]
        self.action_map[interaction_spec['type']](interaction_spec)
        return np.zeros(8), True # Return true to go home

    def predict_action(self, obs: dict, gt_preset: dict = None) -> dict:
        flow = obs['flow']
        img = obs['img']

        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        if img.shape[1] != 3:
            img = einops.rearrange(img, "b h w c -> b c h w")
        assert img.ndim == 4 and img.shape[1] == 3, "Input observation should be in (B, C, H, W) format with 3 channels."

        # flow should be relative, normalized, shape b c t h w
        if flow.ndim != 5: 
            flow = np.expand_dims(flow, axis=0)
        assert flow.ndim == 5 
        b, c, t, h, w = flow.shape
        assert h == 20 and w == 20 and c == 2, "Expected input shape (b, 2, t, 20, 20)"

        # Rearrange into patches: (b, 2, t, 2* 10, 2* 10) → (b, 4, 2, t, 10, 10)
        flow = flow / 200
        flow_patches = einops.rearrange(flow, 'b c t (ph p1) (pw p2) -> b (ph pw) c t p1 p2', ph=2, pw=2)
        flow_flat = einops.rearrange(flow_patches, 'b n c t p1 p2 -> b n (c t p1 p2)')
        flow_tokens = self.flow_proj(torch.tensor(flow_flat).to(self.device).float())  # b, 4, token_dim

        img = img / 255
        encoder_feat = self.encoder(torch.tensor(img).to(self.device)).squeeze()
        img_tokens = self.img_proj(encoder_feat)  # b, 1, token_dim
        if len(img_tokens.shape) == 1:
            img_tokens = img_tokens.unsqueeze(0)
        img_tokens = img_tokens.unsqueeze(1)

        action_token = self.action_token.repeat(b, 1, 1) # b, 1, token_dim


        tokens = torch.cat([flow_tokens, img_tokens, action_token], dim=1) 
        pred_tokens = self.transformer(tokens)
        action = self.action_projector(pred_tokens[:, -1, :])
        type = self.type_projector(pred_tokens[:, -1, :])

        loss = None
        mse = None
        cross_entropy = None
        if gt_preset is not None:
            gt_type = gt_preset['type']
            gt_action = np.concatenate((
                np.array(gt_preset['start_state']),
                np.array(gt_preset['end_state']),
                np.expand_dims(np.array(gt_preset['rotation']), axis=1)
            ), axis=1)

            gt_action_tensor = torch.tensor(gt_action, dtype=torch.float32, device=self.device)
            mse = nn.functional.mse_loss(action, gt_action_tensor)

            gt_type_tensor = torch.tensor(gt_type, dtype=torch.float32, device=self.device)
            cross_entropy = nn.functional.cross_entropy(type, gt_type_tensor)

            loss = mse * 5 + cross_entropy
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return {
            'type': torch.argmax(type, dim=-1),
            'start_state': action[:, :3].cpu().detach().numpy().squeeze(),
            'end_state': action[:, 3:6].cpu().detach().numpy().squeeze(),
            'rotation': action[:, -1].cpu().detach().numpy().squeeze(),
            'loss': loss,
            'mse': mse,
            'cross_entropy': cross_entropy,
        }          

    def get_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dict = self.predict_action(batch['obs'], batch['action'])
        aux_losses = {
            'total_loss': dict['loss'],
            'mse_loss': dict['mse'],
            'cross_entropy_loss': dict['cross_entropy'],
        }
        return dict['loss'], aux_losses

    def step_optimizer(self,
                       global_step: int,
                       batch: dict[str, torch.Tensor],
                       scaler: torch.amp.GradScaler= None) -> dict[str, float]:
        self.optimizer.zero_grad()
        loss, aux_losses = self.get_loss(batch)

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return aux_losses

    def log_ckpt(self, epoch_num: int, type: str) -> None:
        ckpt_path = "ckpt"
        os.makedirs(ckpt_path, exist_ok=True)
        # state_dict = {name: p.data for name, p in self.named_parameters() if p.requires_grad}
        full_state_dict = self.state_dict()
        state_dict = {
            k: v for k, v in full_state_dict.items()
            if not k.startswith("encoder.") or any(p.requires_grad for p in self.encoder.parameters())
        }

        if type == "last":
            torch.save(state_dict, os.path.join(ckpt_path, "model_last.pth"))
        elif type == "log":
            torch.save(state_dict, os.path.join(ckpt_path, f"model_step{epoch_num:05}.ckpt"))
        elif type == "best":
            torch.save(state_dict, os.path.join(ckpt_path, "model_best.pth"))  

