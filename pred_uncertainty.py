import os
import tqdm
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

from utils import freeze_model
  
def check_rank_zero() -> bool:
    return dist.get_rank() == 0 if dist.is_initialized() else True

def train(cfg: DictConfig, local_rank: int) -> None:
    dataset = hydra.utils.instantiate(
        cfg.dataset,
        data_type = cfg.data_type,  # 'intervention' or 'image'
        samplers_per_epoch = cfg.samplers_per_epoch,
    )
    train_sampler = None
    
    model = hydra.utils.instantiate(
        cfg.model,
    ).to(cfg.device)

    if model.encoder is None:
        image_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(cfg.device)
        freeze_model(image_encoder)
        image_encoder.eval()
    else: 
        image_encoder = model.encoder

    print("Model initialized ..")

    if cfg.distributed:
        dist.init_process_group(backend='nccl')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
        num_workers=cfg.num_workers,
    )

    scaler = torch.amp.GradScaler()
    best_eval_loss = float('inf')

    # if cfg.distributed and check_rank_zero():
    cfg_save_path = os.path.join(os.getcwd(), "updated_config.yaml")
    with open(cfg_save_path, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
    wandb.init(project=cfg.model.name)

    for epoch in range(cfg.epochs):
        epoch_loss = []
        if cfg.distributed and train_sampler is not None:
            dataloader.sampler.set_epoch(epoch)
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch}") as pbar:

            for batch in dataloader:
                observations = batch['image'].float().to(cfg.device)
                uncertainties = batch['uncertainty'].float().to(cfg.device)
                text_cond = batch['text_cond'].float().to(cfg.device) if 'text_cond' in batch else None

                if cfg.distributed:
                    aux_losses = model.module.step_optimizer(
                        cfg.device, observations, uncertainties, text_cond=text_cond, scaler=scaler, encoder=image_encoder)
                else:
                    aux_losses = model.step_optimizer(
                        cfg.device, observations, uncertainties, text_cond=text_cond, scaler=scaler, encoder=image_encoder)

                aux_losses = {k: v.item() for k, v in aux_losses.items()}
                epoch_loss.append(aux_losses)
                pbar.set_postfix(loss=aux_losses['total_loss'])
                pbar.update(1)

        epoch_loss = dict(functools.reduce(operator.add, map(collections.Counter, epoch_loss)))
        epoch_loss = {k: v / len(dataloader) for k, v in epoch_loss.items()}

        if check_rank_zero():
            eval_model = model.module if cfg.distributed else model
            if (epoch + 1) % cfg.eval_interval == 0:
                eval_subset = Subset(dataset,
                    indices=torch.randperm(len(dataset))[:cfg.eval_subset_size]
                )
                eval_dataloader = DataLoader(
                    eval_subset,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                )

                epoch_loss['eval_loss'] = eval(epoch=epoch,
                    device=cfg.device,
                    model=eval_model,
                    encoder=image_encoder,
                    dataloader=eval_dataloader)

                if epoch_loss['eval_loss'] < best_eval_loss:
                    best_eval_loss = epoch_loss['eval_loss']
                    eval_model.log_ckpt(epoch, "best")
                
            eval_model.log_ckpt(epoch, "last")
            if (epoch + 1) % cfg.save_interval == 0:
                eval_model.log_ckpt(epoch, "log")

            # if cfg.distributed:
            wandb.log(epoch_loss, step=epoch)

@torch.no_grad()
def eval(epoch: int,
         device: torch.device,
         model: torch.nn.Module,
         encoder: torch.nn.Module,
         dataloader: DataLoader) -> float:
    print(colored("[Eval] ", "green") + f"Evaluating epoch {epoch} ...")
    model.eval()
    eval_loss = []

    for batch in dataloader:
        observations = batch['image'].float().to(device)
        uncertainties = batch['uncertainty'].float().to(device)
        text_cond = batch['text_cond'].float().to(device) if 'text_cond' in batch else None

        loss, aux_losses = model.get_loss(observations, uncertainties, text_cond=text_cond, encoder=encoder)
        eval_loss.append(loss.item())

    eval_loss = sum(eval_loss) / len(eval_loss)
    print(colored("[Eval] ", "green") + f"Epoch {epoch} evaluation loss: {eval_loss:.8f}")
    model.train()
    return eval_loss


@hydra.main(version_base="1.1", config_path="conf", config_name="pred_uncertainty")
def main(cfg: DictConfig) -> None:
    if cfg.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if not check_rank_zero():
            cfg.output_dir = os.devnull
    else:
        local_rank = 0
    train(cfg, local_rank=local_rank)

if __name__ == "__main__":
    main()