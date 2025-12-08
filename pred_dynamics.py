import os
import sys
import tqdm
import copy
import yaml
import hydra
import torch
import wandb
import einops 
import collections, functools, operator
from termcolor import colored
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from model import unpatchify
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def check_rank_zero() -> bool:
    return dist.get_rank() == 0 if dist.is_initialized() else True

def train(cfg: DictConfig, local_rank: int) -> None:
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_sampler = None

    model = hydra.utils.instantiate(cfg.model).to(cfg.device)

    if cfg.distributed:
        dist.init_process_group(backend='nccl')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
        num_workers=cfg.num_workers,
    )
    print(colored("[DataLoader] ", "green") + f"Loaded dataset with {len(dataset)} samples.")

    scaler = torch.amp.GradScaler()
    best_eval_loss = float('inf')

    if cfg.distributed and check_rank_zero():
        cfg_save_path = os.path.join(os.getcwd(), "updated_config.yaml")
        with open(cfg_save_path, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
        wandb.init(project=cfg.model.name)

    for epoch in range(cfg.epochs):
        epoch_loss_list = []
        if cfg.distributed:
            dataloader.sampler.set_epoch(epoch)
        with tqdm.tqdm(total = len(dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(dataloader):

                if cfg.distributed:
                    aux_losses = model.module.step_optimizer(
                        batch=batch,
                        scaler=scaler,
                    )
                else:
                    aux_losses = model.step_optimizer(
                        batch=batch,
                        scaler=scaler,
                    )

                aux_losses = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in aux_losses.items()}
                epoch_loss_list.append(aux_losses)
                pbar.set_postfix(loss=aux_losses["total_loss"])
                pbar.update(1)
        epoch_loss = dict(functools.reduce(operator.add, map(collections.Counter, epoch_loss_list)))
        epoch_loss = {k: v / len(dataloader) for k, v in epoch_loss.items()}

        if check_rank_zero():

            eval_model = model.module if cfg.distributed else model
            if (epoch + 1) % cfg.eval_interval == 0:
                eval_dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    sampler=train_sampler,
                    shuffle=True if train_sampler is None else False,
                    num_workers=cfg.num_workers,
                )

                epoch_loss['eval_loss'] = eval(epoch, cfg, cfg.device, eval_model, eval_dataloader)
                if epoch_loss['eval_loss'] < best_eval_loss:
                    best_eval_loss = epoch_loss['eval_loss']
                    eval_model.log_ckpt(epoch, "best")

                eval_model.log_ckpt(epoch, "last")
                if (epoch + 1) % cfg.save_interval == 0:
                    eval_model.log_ckpt(epoch, "log")

            if cfg.distributed:
                wandb.log(epoch_loss, step=epoch)

@torch.no_grad()
def eval(epoch: int,
         cfg: DictConfig,
         device: torch.device,
         model: torch.nn.Module,
         dataloader: DataLoader = None) -> float:
    print(colored("[Eval] ", "green") + f"Evaluating epoch {epoch}...")
    model.eval()
    eval_loss = []
    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):

        results = model(batch['obs'], batch['action'])

        goal_obs = torch.tensor(batch["obs"]["goal_img"]).to(device) / 255
        if goal_obs.shape[-1] != 3:
            goal_obs = einops.rearrange(goal_obs, 'b c h w -> b h w c')
        if len(goal_obs.shape) == 4:
            goal_obs = einops.repeat(goal_obs, 'b h w c -> b t h w c', t=results.shape[1])
        assert len(goal_obs.shape) == 5 and goal_obs.shape[-1] == 3, "Input videos should be of shape (B, H, W, C)"

        # Compute loss
        mse_loss = ((goal_obs - results) ** 2).mean()
        eval_loss.append(mse_loss)
        model.log_images(batch, results, "eval", epoch)
        del batch, results

    eval_loss = sum(eval_loss) / len(eval_loss)
    print(colored("[Eval] ", "green") + f"Epoch {epoch} evaluation loss: {eval_loss:.4f}")
    model.train()  
    return eval_loss

@hydra.main(version_base="1.1", config_path="conf", config_name="pred_dynamics")
def main(cfg: DictConfig) -> None:
    # set_seed()
    if cfg.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
    train(cfg, local_rank = local_rank)

if __name__ == "__main__":
    if dist.is_initialized() and dist.get_rank() != 0: # Just to avoid creating multiple output directories
        # Hijack sys.argv to avoid creating output dir
        sys.argv.append("hydra.run.dir=null")
        sys.argv.append("hydra.output_subdir=null")
    main()
