import os
import torch
import einops
import pickle
import natsort
import numpy as np
from torch import nn
from termcolor import colored
from policy import CorrectionPolicy

class PresetPolicy(CorrectionPolicy):
    def __init__(self, 
                 name: str = "PresetPolicy",
                 open_loop: bool = False,
                 preset_path: str = None,
                 encoder: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 **kwargs) -> None:
        super(PresetPolicy, self).__init__(**kwargs)
        self.open_loop = open_loop
        self.action_map = {
            'pull': self._pull,
            'pick': self._pick,
            'rotate': self._rotate
        }
        self.types = ['pull', 'pick', 'rotate']
        if open_loop: # Only when non-parameterized policy
            assert preset_path is not None, "Preset path must be provided for open loop preset policy."
            with open(preset_path, 'rb') as f:
                self.preset_data = pickle.load(f)
            print(colored("[PresetPolicy] ", "green") + f"Using Openloop Preset Policy. Loaded preset files from {preset_path}") # preset_path should be a pkl file
        else:
            assert encoder is not None and optimizer is not None, "Encoder and optimizer must be provided for parameterized preset policy."
            self.encoder = encoder
            if any(p.requires_grad for p in self.encoder.parameters()): # Using resnet
                self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final fc layer

            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                encoder_out_dim = self.encoder(dummy_input).squeeze().shape[0]

            self.action_projector = nn.Sequential(
                 nn.Linear(encoder_out_dim, 512),
                 nn.ReLU(),
                 nn.Linear(512, 256),
                 nn.ReLU(),
                 nn.Linear(256, 64),
                 nn.ReLU(),
                 nn.Linear(64, 7)  # Start state (3) + end state (3) + rotation (1)
            )

            self.type_projector = nn.Sequential(
                nn.Linear(encoder_out_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # 3 logits: pull, pick, rotate
            )

            self.trainable_params = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = optimizer(params=self.trainable_params)
            print(colored("[PresetPolicy] ", "green") + "Using Parameterized Preset Policy.")
            
    # def init_robot(self, robot: FrankaResearch3 = None, conn = None, conn_gripper = None) -> None:
    #     self.robot = robot
    #     self.conn = conn
    #     self.conn_gripper = conn_gripper
    
    @torch.no_grad()
    def get_action(self, obs_dict: dict[str, np.array]) -> tuple[np.ndarray, bool]:
        assert self.robot is not None, "Robot must be provided for PresetPolicy" # NOTE: Because of positional control
        obs = obs_dict['image'][-1]
        flow, uncertainty = self._get_flow_and_uncertainty(obs)
        print(colored("[PresetPolicy] ", "green") + f"Current uncertainty is: {uncertainty}")
        # return np.zeros(8), True
        if uncertainty <= self.uncertainty_threshold:
            return None, True # Correction finished
        else:
            if self.open_loop:
                interaction_spec = self._open_loop_action(obs)
            else: 
                interaction_spec = self.predict_action(obs)
                interaction_spec['type'] = self.types[interaction_spec['type']]
            print(interaction_spec)
            self.action_map[interaction_spec['type']](interaction_spec)
        return np.zeros(8), True # Return true to go home
    
    def _open_loop_action(self, obs: np.ndarray) -> np.array:
        # NOTE: For now assume this is not a parameterized policy, and get action based on closest init observation
        closest_dist = float('inf')
        closest_spec = None
        for idx, img in enumerate(self.preset_data['img']):
            dist = np.linalg.norm(obs - img)
            if dist < closest_dist:
                closest_dist = dist
                closest_spec = self.preset_data['correction_specs'][idx]
        assert closest_spec is not None, "No closest spec found"
        return closest_spec
    
    def predict_action(self, obs: np.ndarray, gt_preset: dict = None) -> dict:
        if obs.ndim != 4:
            obs = np.expand_dims(obs, axis=0) 
        if obs.shape[1] != 3:
            obs = einops.rearrange(obs, "b h w c -> b c h w")
        assert obs.ndim == 4 and obs.shape[1] == 3, "Input observation has wrong shape!"

        obs = obs / 255 
        encoder_feat = self.encoder(torch.tensor(obs).to(self.device)).squeeze()
        action = self.action_projector(encoder_feat)
        type = self.type_projector(encoder_feat)

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

            gt_action_tensor = torch.tensor(gt_action, dtype=torch.float32, device=self.device) * 10
            mse = nn.functional.mse_loss(action, gt_action_tensor)

            gt_type_tensor = torch.tensor(gt_type, dtype=torch.float32, device=self.device)
            cross_entropy = nn.functional.cross_entropy(type, gt_type_tensor)

            loss = mse * 5 + cross_entropy
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return {
            'type': torch.argmax(type, dim=-1),
            'start_state': action[:, :3].cpu().detach().numpy().squeeze() / 10,
            'end_state': action[:, 3:6].cpu().detach().numpy().squeeze() / 10,
            'rotation': action[:, -1].cpu().detach().numpy().squeeze() / 10,
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
                       scaler: torch.amp.GradScaler = None) -> dict[str, float]:
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


    # def _pull(self, interaction_spec: dict = None) -> None:
    #     self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
    #     start_state = interaction_spec['start_state'] - np.array([0.0, 0.0, 0.02])
    #     self.robot.go2cartesian(self.conn, start_state)
    #     end_state = interaction_spec['end_state']
    #     end_state[2] = interaction_spec['start_state'][2] # z value
    #     self.robot.go2cartesian(self.conn, end_state)
    #     self.robot.go2cartesian(self.conn, end_state + np.array([0.0, 0.0, 0.05]))

    # def _pick(self, interaction_spec: dict = None) -> None:
    #     self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
    #     start_state = interaction_spec['start_state'] - np.array([0.0, 0.0, 0.01])
    #     self.robot.go2cartesian(self.conn, start_state)
    #     self.robot.send2gripper(self.conn_gripper, "c")
    #     time.sleep(1)

    #     up_start_state = interaction_spec['start_state'] + np.array([0.0, 0.0, 0.1])
    #     up_end_state = interaction_spec['end_state'] + np.array([0.0, 0.0, 0.1])
    #     self.robot.go2cartesian(self.conn, up_start_state)
    #     self.robot.go2cartesian(self.conn, up_end_state)

    #     self.robot.go2cartesian(self.conn, interaction_spec['end_state'])
    #     self.robot.send2gripper(self.conn_gripper, "o")

    #     self.robot.go2cartesian(self.conn, up_end_state)

    # def _rotate(self, interaction_spec: dict = None) -> None:
    #     self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
    #     start_state = interaction_spec['start_state'] - np.array([0.0, 0.0, 0.0])
    #     self.robot.go2cartesian(self.conn, start_state)
    #     self.robot.send2gripper(self.conn_gripper, "c")
    #     time.sleep(1)
    #     self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
    #     time.sleep(1)

    #     curr_joint_state = self.robot.readState(self.conn)
    #     rotate_joint_state = curr_joint_state["q"].copy()
    #     rotate_joint_state[-1] += interaction_spec['rotation']  # Rotate the last joint
    #     self.robot.go2position(self.conn, list(rotate_joint_state))
    #     self.robot.send2gripper(self.conn_gripper, "o")
        
    #     curr_joint_state = self.robot.readState(self.conn)["q"]
    #     curr_cart_state = self.robot.joint2pose(curr_joint_state)[0]
    #     self.robot.go2cartesian(self.conn, curr_cart_state + np.array([0.0, 0.0, 0.05]))

        
