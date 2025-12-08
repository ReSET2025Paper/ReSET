import os
import time
import hydra
import torch
import numpy as np
import torch.nn as nn

from termcolor import colored
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"

from utils import FrankaResearch3

class Policy(nn.Module):
    """
    General class for policies.
    """

    def __init__(self, device: str, action_scale: float = 14, **kwargs) -> None:
        """
        Initializes the Policy with a specific task.
        """
        super(Policy, self).__init__()
        self.device = device
        self.action_scale = action_scale
    
    def get_action(self):
        raise NotImplementedError
    

class CorrectionPolicy(Policy):
    """
    General class for corrections policies.
    """

    def __init__(self,
                 uncertainty_threshold: float,
                 visualize_path: str = None,
                 conditioned: bool = False,
                 **kwargs) -> None:
        super(CorrectionPolicy, self).__init__(**kwargs)
        """
        Initializes the CorrectionPolicy with a specific task.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.visualize_path = visualize_path
        self.conditioned = conditioned

    def initialize_models(self, flow_model_path: str, uncertainty_model_path: str) -> None:
        if flow_model_path is not None:
            if os.path.exists(f"{flow_model_path}/updated_config.yaml"):
                flow_cfg = OmegaConf.load(f"{flow_model_path}/updated_config.yaml")
            else:
                flow_cfg = OmegaConf.load(f"{flow_model_path}/.hydra/config.yaml")
            self.flow_model = hydra.utils.instantiate(flow_cfg.model).to(self.device)
            self.flow_model.load_state_dict(torch.load(f"{flow_model_path}/ckpt/model_best.pth", weights_only=True), strict=True)
            self.flow_model.eval()

            if "min_flow" not in flow_cfg:
                print(colored("[Warning] ", "yellow") + "min_flow not found in flow_cfg, using default value.")
            self.min_flow = np.array(flow_cfg.get("min_flow", [-66.94593811035156, -59.26799392700195]))
            self.max_flow = np.array(flow_cfg.get("max_flow", [301.9591979980469, 220.32180786132812]))
        else:
            self.flow_model = None

        if os.path.exists(f"{uncertainty_model_path}/updated_config.yaml"):
            uncertainty_cfg = OmegaConf.load(f"{uncertainty_model_path}/updated_config.yaml")
        else:
            uncertainty_cfg = OmegaConf.load(f"{uncertainty_model_path}/.hydra/config.yaml")
        self.uncertainty_model = hydra.utils.instantiate(uncertainty_cfg.model).to(self.device)            
        self.uncertainty_model.load_state_dict(torch.load(f"{uncertainty_model_path}/ckpt/model_best.pth", weights_only=True), strict=True)
        self.uncertainty_model.eval()

        self.dino_encoder = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14_reg').to(self.device).eval()
        print(colored("[Correction Policy] ", "green") + "Correction base policy initialized!")

        if self.conditioned: 
            assert self.flow_model.conditioned and self.uncertainty_model.conditioned, "Models should be conditioned for text input"
            self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.okenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            

        
    def init_robot(self, robot: FrankaResearch3 = None, conn = None, conn_gripper = None) -> None:
        self.robot = robot
        self.conn = conn
        self.conn_gripper = conn_gripper

        
    def _get_flow_and_uncertainty(self, obs: np.array, text_cond: str = None):  # obs shape: (B, C, H, W), normalized
        try:
            assert obs.ndim == 4 and obs.shape[1] == 3
        except AssertionError as e:
            if obs.ndim == 3:
                obs = np.expand_dims(obs, axis = 0)
            if obs.shape[-1] == 3:
                obs = np.moveaxis(obs, -1, 1)
        assert obs.ndim == 4 and obs.shape[1] == 3, "Observation should be a 4D array (B, C, H, W)"

        if text_cond is not None:
            assert self.uncertainty_model.conditioned and self.flow_model.conditioned, "Models should be conditioned for text input"
            inputs = self.tokenizer(text_cond, return_tensors="pt", truncation=True, padding=True).to(next(self.text_encoder.parameters()).device)
            with torch.no_grad():
                text_cond = self.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze()

        obs = torch.tensor(obs / 255).to(self.device)

        with torch.no_grad():
            # Uncertainty estimation
            if self.uncertainty_model.encoder is None:
                feat = self.dino_encoder.forward_features(obs)['x_norm_patchtokens']
            else:
                feat = self.uncertainty_model.encoder(obs)
            uncertainty = self.uncertainty_model(feat, text_cond=text_cond)

            # Flow prediction
            if self.flow_model is None:
                flow = None
            else:
                if self.flow_model.encoder is None:
                    flow_feat = self.dino_encoder.forward_features(obs)['x_norm_patchtokens']
                else:
                    flow_feat = self.flow_model.encoder(obs)
                flow = self.flow_model(flow_feat, text_cond=text_cond)
                flow = flow.squeeze(0).detach().cpu().numpy()

        # flow shape: (B, C, T, H, W) 
        return flow, uncertainty.item()
    
    def _pull(self, interaction_spec: dict = None) -> None:
        self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
        start_state = interaction_spec['start_state'] - np.array([0.0, 0.0, 0.02])
        self.robot.go2cartesian(self.conn, start_state)
        end_state = interaction_spec['end_state']
        end_state[2] = interaction_spec['start_state'][2] # z value
        self.robot.go2cartesian(self.conn, end_state)
        self.robot.go2cartesian(self.conn, end_state + np.array([0.0, 0.0, 0.05]))

    def _pick(self, interaction_spec: dict = None) -> None:
        self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
        start_state = interaction_spec['start_state'] - np.array([0.0, 0.0, 0.02])
        self.robot.go2cartesian(self.conn, start_state)
        self.robot.send2gripper(self.conn_gripper, "c")
        time.sleep(1)

        up_start_state = interaction_spec['start_state'] + np.array([0.0, 0.0, 0.1])
        up_end_state = interaction_spec['end_state'] + np.array([0.0, 0.0, 0.1])
        self.robot.go2cartesian(self.conn, up_start_state)
        self.robot.go2cartesian(self.conn, up_end_state)

        self.robot.go2cartesian(self.conn, interaction_spec['end_state'] - np.array([0.0, 0.0, 0.12]))
        self.robot.send2gripper(self.conn_gripper, "o")

        self.robot.go2cartesian(self.conn, up_end_state)

    def _rotate(self, interaction_spec: dict = None) -> None:
        self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
        start_state = interaction_spec['start_state'] - np.array([0.0, 0.0, 0.01])
        self.robot.go2cartesian(self.conn, start_state)
        self.robot.send2gripper(self.conn_gripper, "c")
        time.sleep(1)
        self.robot.go2cartesian(self.conn, interaction_spec['start_state'] + np.array([0.0, 0.0, 0.05]))
        time.sleep(1)

        curr_joint_state = self.robot.readState(self.conn)
        rotate_joint_state = curr_joint_state["q"].copy()
        rotate_joint_state[-1] += interaction_spec['rotation']  # Rotate the last joint
        self.robot.go2position(self.conn, list(rotate_joint_state))
        
        curr_joint_state = self.robot.readState(self.conn)["q"]
        curr_cart_state = self.robot.joint2pose(curr_joint_state)[0]
        self.robot.go2cartesian(self.conn, curr_cart_state - np.array([0.0, 0.0, 0.04]))
        time.sleep(1)
        self.robot.send2gripper(self.conn_gripper, "o")
        time.sleep(1)
        self.robot.go2cartesian(self.conn, curr_cart_state + np.array([0.0, 0.0, 0.05]))


