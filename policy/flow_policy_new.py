import os
import torch
import einops
import numpy as np
from torch import nn
from termcolor import colored
from policy import CorrectionPolicy

import cv2
from utils import visualize_pred_tracking

class FlowPolicyNew(CorrectionPolicy):
    def __init__(self,
                 name: str = "FlowPolicy",
                 encoder: torch.nn.Module = None,
                 flow_steps: int = 18,
                 flow_shape: tuple = (20, 20),
                 token_dim: int = 128,
                 optimizer: torch.optim.Optimizer = None,
                 **kwargs) -> None:
        super(FlowPolicyNew, self).__init__(**kwargs)
        # self.flow_proj = nn.Sequential(
        #     nn.Linear(2 * (flow_steps - 1) * flow_shape[0] // 2 * flow_shape[1] // 2, 256), # -1 for relative flow
        #     nn.ReLU(),
        #     nn.Linear(256, token_dim)
        # )
        self.flow_proj = nn.Sequential(
            nn.Linear(3 * (flow_steps - 1), 256),  # 3 * (flow_steps - 1)
            nn.ReLU(),
            nn.Linear(256, token_dim),
        )
        
        # Image encoder
        assert encoder is not None and optimizer is not None, "Encoder and optimizer must be provided for parameterized preset policy."
        self.encoder = encoder
        dummy_input = torch.randn(1, 3, 224, 224)
        if any(p.requires_grad for p in self.encoder.parameters()): # Using resnet
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final fc layer
            with torch.no_grad():
                encoder_out_dim = self.encoder(dummy_input).squeeze().shape[0]
        for p in self.encoder.parameters():
            p.requires_grad = False
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
        # ========= 0) 取输入并基础检查 =========
        flow = obs['flow']
        img = obs['img']

        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        if img.shape[1] != 3:
            img = einops.rearrange(img, "b h w c -> b c h w")
        assert img.ndim == 4 and img.shape[1] == 3, "Input observation should be in (B, C, H, W) format with 3 channels."

        if flow.ndim != 5:
            flow = np.expand_dims(flow, axis=0)
        assert flow.ndim == 5
        b, c, t, h, w = flow.shape
        assert (h, w, c) == (20, 20, 2), "Expected input shape (b, 2, t, 20, 20)"

        # ========= 1) 预处理：稳噪&方向优先的 flow token =========
        # 归一化到可控量级
        flow = flow / 200.0  # [像素]到较小范围

        # (a) 方向/幅值分解：方向更稳定（你说“方向应相似”）
        # flow_vec: (b, 2, t, h, w)
        eps = 1e-6
        speed = np.sqrt((flow[:, 0] ** 2 + flow[:, 1] ** 2))                       # (b, t, h, w)
        unit_x = flow[:, 0] / (speed + eps)                                        # (b, t, h, w)
        unit_y = flow[:, 1] / (speed + eps)
        log_speed = np.log(speed + eps)                                            # 强度用对数更稳

        # 把 (unit_x, unit_y, log_speed) 拼一起作为 3 通道，再做 2x2 patch
        flow_feat = np.stack([unit_x, unit_y, log_speed], axis=1)                  # (b, 3, t, h, w)

        # 粗池化抑噪 + token 化（2x2）
        flow_patches = einops.rearrange(
            flow_feat, 'b c t (ph p1) (pw p2) -> b (ph pw) c t p1 p2', ph=2, pw=2
        )  # (b, 4, 3, t, 2, 2)

        # 先做小区域平均，进一步抑噪
        flow_patches = flow_patches.mean(axis=(-1, -2))                            # (b, 4, 3, t)

        # 拉平为 token 输入
        flow_flat = einops.rearrange(flow_patches, 'b n c t -> b n (c t)')         # (b, 4, 3*t)
        flow_tokens = self.flow_proj(torch.tensor(flow_flat, device=self.device).float())  # (b, 4, token_dim)

        # ========= 2) 图像编码：避免 squeeze 掉 batch =========
        img_t = torch.tensor(img, device=self.device).float() / 255.0
        enc = self.encoder(img_t)
        # enc 可能是 (b, C, H', W') 或 (b, C, 1, 1) 或 (b, C)
        if enc.dim() == 4:
            encoder_feat = enc.mean(dim=(2, 3))     # GAP -> (b, C)
        elif enc.dim() == 2:
            encoder_feat = enc                      # (b, C)
        else:
            encoder_feat = enc.view(enc.size(0), -1)
        img_tokens = self.img_proj(encoder_feat).unsqueeze(1)  # (b, 1, token_dim)

        # ========= 3) 模态 Dropout（训练时更依赖 flow）=========
        if self.training and torch.rand(1, device=self.device).item() < 0.5:
            img_tokens = torch.zeros_like(img_tokens)

        # ========= 4) 主前向：flow + image + action token =========
        action_token = self.action_token.repeat(b, 1, 1)  # (b, 1, token_dim)
        tokens = torch.cat([flow_tokens, img_tokens, action_token], dim=1)  # (b, 4+1+1, d)
        pred_tokens = self.transformer(tokens)
        z = pred_tokens[:, -1, :]                          # (b, d)
        action = self.action_projector(z)                  # (b, 7)
        type_logits = self.type_projector(z)               # (b, 3)

        # ========= 5) flow-only 辅助前向：强制从 flow 学 =========
        # 用同一 transformer，但把 img token 置零，做第二次前向，产生辅助监督
        tokens_flow_only = torch.cat([flow_tokens,
                                    torch.zeros_like(img_tokens),
                                    action_token], dim=1)
        pred_tokens_flow = self.transformer(tokens_flow_only)
        z_flow = pred_tokens_flow[:, -1, :]
        action_flow = self.action_projector(z_flow)        # (b, 7)
        type_logits_flow = self.type_projector(z_flow)     # (b, 3)

        # ========= 6) 计算损失（含修正）=========
        loss = mse = cross_entropy = None
        if gt_preset is not None:
            # --- 动作 GT ---
            gt_action = np.concatenate((
                np.array(gt_preset['start_state']),
                np.array(gt_preset['end_state']),
                np.expand_dims(np.array(gt_preset['rotation']), axis=1)
            ), axis=1)
            gt_action_tensor = torch.tensor(gt_action, dtype=torch.float32, device=self.device) * 10.0

            num_classes = 3
            raw_type = gt_preset['type']

            # to tensor (don’t re-wrap a tensor with torch.tensor)
            if torch.is_tensor(raw_type):
                tgt = raw_type.to(self.device)
            else:
                tgt = torch.from_numpy(np.asarray(raw_type)).to(self.device)

            # If one-hot / probs: (B, C) -> indices (B,)
            if tgt.dim() == 2 and tgt.size(-1) == num_classes:
                # if it’s float probs already, you could keep it as float and pass directly to CE,
                # but we’ll convert to indices for clarity/stability:
                gt_type_tensor = tgt.argmax(dim=-1).long()
            else:
                # assume it’s already indices shape (B,)
                gt_type_tensor = tgt.long()

            # 主损失
            mse = nn.functional.mse_loss(action, gt_action_tensor)
            cross_entropy = nn.functional.cross_entropy(type_logits, gt_type_tensor)

            # flow-only 辅助损失（逼模型从 flow 分支学）
            mse_aux = nn.functional.mse_loss(action_flow, gt_action_tensor)
            ce_aux = nn.functional.cross_entropy(type_logits_flow, gt_type_tensor)

            # 方向对齐（主分支 + 浅权重）：让预测位移方向与 GT 对齐（小数据稳）
            start = torch.tensor(gt_preset['start_state'], dtype=torch.float32, device=self.device)
            end   = torch.tensor(gt_preset['end_state'],   dtype=torch.float32, device=self.device)
            delta = (end - start)                                   # (b, 3)
            u_gt = delta / (delta.norm(dim=1, keepdim=True) + 1e-6)

            # 从 action 里取预测位移并归一
            u_pred = (action[:, 3:6] - action[:, :3])
            u_pred = u_pred / (u_pred.norm(dim=1, keepdim=True) + 1e-6)
            L_dir = 1.0 - (u_pred * u_gt).sum(dim=1).mean()

            # rotation 用圆周损失更稳（可选，沿用原 7 维 MSE 也行）
            theta_pred = action[:, -1]
            theta_gt   = gt_action_tensor[:, -1]
            L_rot = 1.0 - torch.cos(theta_pred - theta_gt).mean()

            # 权重（可调）
            w_mse, w_ce = 5.0, 1.0
            w_aux_mse, w_aux_ce = 0.5, 0.5
            w_dir, w_rot = 0.5, 0.5

            loss = (w_mse * mse + w_ce * cross_entropy
                    + w_aux_mse * mse_aux + w_aux_ce * ce_aux
                    + w_dir * L_dir + w_rot * L_rot)

        # 保证形状
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return {
            'type': torch.argmax(type_logits, dim=-1),
            'start_state': action[:, :3].detach().cpu().numpy().squeeze() / 10.0,
            'end_state':   action[:, 3:6].detach().cpu().numpy().squeeze() / 10.0,
            'rotation':    action[:, -1].detach().cpu().numpy().squeeze() / 10.0,
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
