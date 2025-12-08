"""Code taken from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py#L10
    @article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
"""
import os
import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from termcolor import colored
from diffusers import DDPMScheduler
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)
from torchvision.models import resnet50, ResNet50_Weights

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

from . import Policy
from model import TransformerForDiffusion, LinearNormalizer, dict_apply, replace_submodules

class DiffusionPolicy(Policy):
    def __init__(self, 
                name: str,
                shape_meta: dict,
                # task params
                horizon, 
                n_action_steps, 
                n_obs_steps,
                num_inference_steps=None,
                # image
                crop_shape=(76, 76),
                obs_encoder_group_norm=False,
                eval_fixed_crop=False,
                # arch
                n_layer=8,
                n_cond_layers=0,
                n_head=4,
                n_emb=256,
                p_drop_emb=0.0,
                p_drop_attn=0.3,
                causal_attn=True,
                time_as_cond=True,
                obs_as_cond=True,
                pred_action_steps_only = True, # Check this
                # optimizer
                optimizer: torch.optim.Optimizer = None,
                transformer_weight_decay: float = 0.0, # Placeholder
                obs_encoder_weight_decay: float = 0.0, # Placeholder
                # normalizer
                normalizer: LinearNormalizer = None,
                # scheduler
                lr_scheduler: dict = None,
                noise_scheduler: DDPMScheduler = None,
                **kwargs) -> None:
        super(DiffusionPolicy, self).__init__(**kwargs)

        # self.obs_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        ### Getting obs encoder from robomimic
        obs_config = {
            'low_dim': [],
            'rgb': []
        }
        obs_key_shapes = dict()
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        config = config_factory(algo_name="bc")
        config = modify_config_for_default_image_exp(config)
        config = modify_config_for_dataset(
            config = config,
            task_name = 'square',
            dataset_type = 'ph',
            hdf5_type = 'image',
            base_dataset_dir = '/tmp/null',
            filter_key = None
        )
        algo_config_modifier = getattr(gpc, f'modify_bc_rnn_config_for_dataset')
        config = algo_config_modifier(
            config=config,
            task_name='square',
            dataset_type='ph',
            hdf5_type='image',
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality['obs_randomizer_class'] = None

        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=shape_meta['action']['shape'][0],
                device='cpu',
            )

        self.obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        ###

        # replace batch norm with group norm:
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

        self.horizon = horizon
        # self.obs_feature_dim = obs_feature_dim
        self.obs_feature_dim = self.obs_encoder.output_shape()[0]
        self.action_dim = shape_meta['action']['shape'][0]
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.num_inference_steps = num_inference_steps

        self.model = TransformerForDiffusion(
            input_dim = self.action_dim,
            output_dim = self.action_dim,
            horizon = self.horizon,
            n_obs_steps = self.n_obs_steps,
            cond_dim = self.obs_feature_dim, # This is assuming obs_as_cond == True
            n_layer = n_layer,
            n_head = n_head,
            n_emb = n_emb,
            p_drop_emb = p_drop_emb,
            p_drop_attn = p_drop_attn,
            causal_attn= causal_attn,
            time_as_cond = time_as_cond,
            obs_as_cond = obs_as_cond,
            n_cond_layers = n_cond_layers
        )

        # set optimizer
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append(
            {
                'params': self.obs_encoder.parameters(),
                'weight_decay': obs_encoder_weight_decay,
            })
        self.optimizer = optimizer(params=optim_groups)
        self.gradient_accumulate_every = lr_scheduler.gradient_accumulate_every if lr_scheduler is not None else 1
        # set normalizer
        self.normalizer = LinearNormalizer()
        if normalizer is None:
            print(colored("[Warning@DiffusionPolicy] ", "yellow") + "No normalizer provided, make sure it will be loaded later.")
        else:
            self.normalizer.load_state_dict(normalizer.state_dict())
        # set noise scheduler
        self.noise_scheduler = noise_scheduler
        # set lr scheduler
        self.lr_scheduler = None
        if lr_scheduler is not None:
            name = SchedulerType(lr_scheduler.name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            self.lr_scheduler = schedule_func(
                optimizer=self.optimizer,
                num_warmup_steps=lr_scheduler.lr_warmup_steps,
                num_training_steps=(
                    lr_scheduler.len_dataloader * lr_scheduler.num_epochs // lr_scheduler.gradient_accumulate_every),
                last_epoch = -1 # assuming no previous training
                )

    # training
    def get_loss(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]: 
        for key, value in batch["obs"].items():
            batch["obs"][key] = value.to(self.device)
        nobs = self.normalizer.normalize(batch['obs']) # dict
        nactions = self.normalizer['action'].normalize(batch['action'].to(self.device)) # tensor
        B, T = nactions.shape[0], nactions.shape[1]
        To = self.n_obs_steps

        this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        cond = nobs_features.reshape(B, To, -1) # (B, To, obs_feature_dim = 136)

        start = To - 1 # self.pred_action_steps_only == True
        end = start + self.n_action_steps
        trajectory = nactions[:, start:end, ...]
        # condition_mask = torch.zeros_like(trajectory, dtype=torch.bool) # self.pred_action_steps_only == True
        # loss_mask = ~condition_mask

        # noise to be added to the trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        pred = self.model(noisy_trajectory, timesteps, cond)

        
        pred_type = self.noise_scheduler.config.prediction_type
        target = noise if pred_type == "epsilon" else trajectory

        loss = F.mse_loss(pred, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        aux_losses = {
            'total_loss': loss,
            'mse_loss': loss,
        }
        return loss, aux_losses
    
    def step_optimizer(self,
                       global_step: int,
                       batch: dict[str, torch.Tensor],
                       scaler: torch.amp.GradScaler = None) -> dict[str, float]:
        self.optimizer.zero_grad()
        raw_loss, aux_losses = self.get_loss(batch)
        loss = raw_loss / self.gradient_accumulate_every
        if scaler is not None:
            scaler.scale(loss).backward()
            if global_step % self.gradient_accumulate_every == 0:
                scaler.step(self.optimizer)
                scaler.update()
        else:
            loss.backward()
            if global_step % self.gradient_accumulate_every == 0:
                if self.obs_encoder.weight_decay > 0:
                    self.optimizer.step()

        if self.lr_scheduler is not None:
            if global_step % self.gradient_accumulate_every == 0:
                self.lr_scheduler.step()
            aux_losses['learning_rate'] = self.lr_scheduler.get_last_lr()[0]

        return aux_losses
    
    # eval
    def conditional_sample(self, 
            condition_data: torch.Tensor, 
            condition_mask: torch.Tensor,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ) -> torch.Tensor:

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # 1. apply conditioning
            # trajectory[condition_mask] = condition_data[condition_mask] 
            # # Not doing this in training, pred_action_steps_only == True

            # 2. predict model output
            model_output = self.model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def predict_action(self, obs_dict: dict[str, torch.Tensor], gt_action = None) -> dict [str, torch.Tensor]:
        for key, value in obs_dict.items():
            obs_dict[key] = value.to(self.device)
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0] # Since not sure which key to use
        To = self.n_obs_steps

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) #obs_as_cond == True
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, To, -1)

        shape = (B, self.n_action_steps, self.action_dim) # self.pred_action_steps_only == True
        cond_data = torch.zeros(size=shape, device=self.device)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=cond
        )
        naction_pred = nsample[...,:self.action_dim] # self.pred_action_steps_only == True
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        mse = None
        if gt_action is not None:
            gt_action = gt_action[:,self.n_obs_steps-1:self.n_obs_steps+self.n_action_steps-1,...].to(self.device)
            mse = F.mse_loss(action_pred, gt_action)

        return {'action': action_pred, 'loss': mse}

    @torch.no_grad()
    def get_action(self, obs_dict: dict[str, np.array]) -> np.array:
        obs_dict = {k: torch.tensor(v, device=self.device) for k, v in obs_dict.items()}
        for k, v in obs_dict.items():
            if k == "image" or "image_gripper":
                if v.shape[-1] == 3:
                    obs_dict[k] = einops.rearrange(v, "T H W C -> T C H W")
            obs_dict[k] = obs_dict[k].unsqueeze(0)
        action_dict = self.predict_action(obs_dict)
        action = action_dict['action'].squeeze(0).detach().cpu().numpy()
        return np.concatenate((action[0, :-1] * self.action_scale, action[0, -1:]))
    
    # save
    def log_ckpt(self, epoch_num: int, type: str) -> None:
        ckpt_path = "ckpt"
        os.makedirs(ckpt_path, exist_ok=True)

        if type == "last":
            torch.save(self.state_dict(), os.path.join(ckpt_path, "model_last.pth"))
        elif type == "log":
            torch.save(self.state_dict(), os.path.join(ckpt_path, f"model_step{epoch_num:05}.ckpt"))
        elif type == "best":
            torch.save(self.state_dict(), os.path.join(ckpt_path, "model_best.pth"))





