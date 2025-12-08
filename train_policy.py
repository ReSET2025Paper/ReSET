
import os
import sys
import tqdm
import copy
import yaml
import hydra
import torch
import wandb
import collections, functools, operator
from termcolor import colored
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def check_rank_zero() -> bool:
    return dist.get_rank() == 0 if dist.is_initialized() else True

def train(cfg: DictConfig, local_rank: int) -> None:
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = dataset.get_normalizer() if hasattr(dataset, 'get_normalizer') else None
    train_sampler = None

    model = hydra.utils.instantiate(cfg.policy, normalizer=normalizer)
    model.to(cfg.device)
    ema_model = copy.deepcopy(model).to(cfg.device) if cfg.use_ema else None

    if cfg.distributed:
        dist.init_process_group(backend='nccl')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if ema_model is not None:
            ema_model = DDP(ema_model, device_ids=[local_rank], output_device=local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )

    ema = None
    if cfg.use_ema:
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
        print(colored("[Policy] ", "green") + "Initialized Exponential Moving Average model.")
    else:
        print(colored("[Policy] ", "yellow") + "Exponential Moving Average model is not used.")

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

    if check_rank_zero():
        cfg_save_path = os.path.join(os.getcwd(), "updated_config.yaml")
        with open(cfg_save_path, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
    # if cfg.distributed and check_rank_zero():
    wandb.init(project=cfg.policy.name)

    for epoch in range(cfg.epochs):
        epoch_loss_list = []
        if cfg.distributed:
            dataloader.sampler.set_epoch(epoch)
        with tqdm.tqdm(total = len(dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(dataloader):

                if cfg.distributed:
                    aux_losses = model.module.step_optimizer(
                        global_step=epoch * len(dataloader) + batch_idx,
                        batch=batch,
                        scaler=scaler,
                    )
                else:
                    aux_losses = model.step_optimizer(
                        global_step=epoch * len(dataloader) + batch_idx,
                        batch=batch,
                        scaler=scaler,
                    )
                if cfg.use_ema:
                    ema.step(model)

                aux_losses = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in aux_losses.items()}
                epoch_loss_list.append(aux_losses)
                pbar.set_postfix(loss=aux_losses["total_loss"])
                pbar.update(1)
        # epoch_loss = dict(functools.reduce(operator.add, map(collections.Counter, epoch_loss_list)))
        epoch_loss = {}
        for d in epoch_loss_list:
            for k, v in d.items():
                epoch_loss[k] = epoch_loss.get(k, 0.0) + v
        if 'learning_rate' in epoch_loss_list[0].keys():
            assert "learning_rate" in epoch_loss.keys(), "Learning rate not found in epoch_loss!!"
        epoch_loss = {k: v / len(dataloader) for k, v in epoch_loss.items()}

        if check_rank_zero():
            if cfg.use_ema:
                policy = ema_model.module if cfg.distributed else ema_model
            else:
                policy = model.module if cfg.distributed else model
            if (epoch + 1) % cfg.eval_interval == 0:
                eval_dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    sampler=train_sampler,
                    shuffle=True if train_sampler is None else False,
                    num_workers=cfg.num_workers,
                )

                epoch_loss['eval_loss'] = eval(epoch, cfg.device, policy, eval_dataloader)
                if epoch_loss['eval_loss'] < best_eval_loss:
                    best_eval_loss = epoch_loss['eval_loss']
                    policy.log_ckpt(epoch, "best")

                policy.log_ckpt(epoch, "last")
                if (epoch + 1) % cfg.save_interval == 0:
                    policy.log_ckpt(epoch, "log")

            # if cfg.distributed:
            wandb.log(epoch_loss, step=epoch)

@torch.no_grad()
def eval(epoch: int,
         device: torch.device,
         policy: torch.nn.Module,
         dataloader: DataLoader = None) -> float:
    print(colored("[Eval] ", "green") + f"Evaluating epoch {epoch}...")
    policy.eval()
    eval_loss = []
    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):

        results = policy.predict_action(batch['obs'], batch['action'])
        eval_loss.append(results["loss"].item())
        del batch, results

    eval_loss = sum(eval_loss) / len(eval_loss)
    print(colored("[Eval] ", "green") + f"Epoch {epoch} evaluation loss: {eval_loss:.4f}")
    policy.train()  
    return eval_loss

@hydra.main(version_base="1.1", config_path="conf", config_name="train_policy")
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
