import os
import hydra
import torch
from typing import Union
from termcolor import colored
from omegaconf import OmegaConf

from policy import Policy

def load_policy(policy_cfg: Union[str,dict], 
                device: str = "cuda", 
                robot = None, 
                conn = None, 
                conn_gripper = None,
                action_scale: float = 14,
                flow_model_path: str = None,
                uncertainty_model_path: str = None,
                uncertainty_threshold: float = None) -> Policy:
    if isinstance(policy_cfg, str):
        if os.path.exists(f"{policy_cfg}/updated_config.yaml"):
            cfg = OmegaConf.load(f"{policy_cfg}/updated_config.yaml")
        else:
            cfg = OmegaConf.load(f"{policy_cfg}/.hydra/config.yaml")
        weight_path = f"{policy_cfg}/ckpt/model_best.pth"
        policy = hydra.utils.instantiate(cfg.policy, 
                                         device = device, 
                                         action_scale = action_scale)
    else:
        cfg = policy_cfg
        weight_path = None
        policy = hydra.utils.instantiate(cfg, 
                                         device = device,  
                                         action_scale = action_scale)

    if weight_path is not None: 
        try:
            policy.load_state_dict(torch.load(weight_path, weights_only=True), strict=True)
        except: 
            print(colored("[Warning] ", "red") + "Failed to load state_dict with strict=True, trying with strict=False. \n Make sure if you are loading a pretrained encoder.")
            policy.load_state_dict(torch.load(weight_path, weights_only=True), strict=False)
    if hasattr(policy, 'init_robot'):
        assert robot is not None, colored("[Error] ", "red") + "Robot must be provided for this policy"
        policy.init_robot(robot, conn, conn_gripper)
    if hasattr(policy, 'initialize_models'): # correction policy
        # assert flow_model_path is not None and uncertainty_model_path is not None, "Flow and uncertainty model paths must be provided for CorrectionPolicy"
        assert uncertainty_model_path is not None, colored("[Error] ", "red") + "Uncertainty model paths must be provided for CorrectionPolicy"
        if flow_model_path is None:
            print(colored("[Warning] ", "yellow") + "Flow model path is not provided, using uncertainty model only.")
        policy.initialize_models(flow_model_path, uncertainty_model_path)
    if hasattr(policy, "uncertainty_threshold"):
        policy.uncertainty_threshold = uncertainty_threshold
        print(colored("[Correction] ", "green") + f"Set uncertainty threshold to be {policy.uncertainty_threshold}")
    policy.eval()

    return policy
