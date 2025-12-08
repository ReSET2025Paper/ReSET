import os
import sys
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

from utils import visualize_pred_tracking, freeze_model, set_seed

def check_rank_zero() -> bool:
    return dist.get_rank() == 0 if dist.is_initialized() else True

def train(cfg: DictConfig, local_rank: int) -> None:
    dataset = hydra.utils.instantiate(
        cfg.dataset,
    )
    cfg.min_flow = [float(x) for x in dataset.data.min_flow]
    cfg.max_flow = [float(x) for x in dataset.data.max_flow]

    train_sampler = None

    model = hydra.utils.instantiate(
        cfg.model,
        # train_encoder=image_encoder
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

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
        num_workers=cfg.num_workers,
    )

    scaler = torch.amp.GradScaler()
    best_eval_loss = float('inf')

    if check_rank_zero():
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
                observations = batch['observations'][:, 0].to(cfg.device)
                tracking = batch['tracking'].to(cfg.device)
                text_cond = batch['text_cond'].float().to(cfg.device) if 'text_cond' in batch else None

                if cfg.distributed:
                    aux_losses = model.module.step_optimizer(
                        cfg.device, observations, tracking, text_cond=text_cond, scaler=scaler, clip_val=cfg.clip_grad_norm, encoder=image_encoder
                    )
                else:
                    aux_losses = model.step_optimizer(
                        cfg.device, observations, tracking, text_cond=text_cond, scaler=scaler, clip_val=cfg.clip_grad_norm, encoder=image_encoder
                    )

                aux_losses = {k: v.detach() for k, v in aux_losses.items()}
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
                    min_flow=dataset.data.min_flow,
                    max_flow=dataset.data.max_flow,
                    device=cfg.device,
                    model=eval_model,
                    encoder=image_encoder,
                    dataloader=eval_dataloader,
                    visualize_dir=cfg.eval_visualize_dir)

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
         device: torch.device,
         min_flow: float,
         max_flow: float,
         model: torch.nn.Module,
         encoder: torch.nn.Module = None,
         dataloader: DataLoader = None,
         visualize_dir: str = None,
) -> float:
    
    print(colored("[Eval] ", "green") + f"Evaluating epoch {epoch} ...")
    model.eval()
    eval_loss = []
    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        input_image = batch['observations'][:, 0].to(device)
        tracking = batch['tracking'].to(device)
        text_cond = batch['text_cond'].float().to(device) if 'text_cond' in batch else None

        loss, aux_losses, pred_tracking = model.get_loss(input_image, tracking, text_cond=text_cond, encoder=encoder)
        eval_loss.append(loss.item())
        # print(pred_tracking.shape, tracking.shape)
        # # Check per-timestep difference in prediction
        # pred_diff = (pred_tracking[:, :, 1:] - pred_tracking[:, :, :-1]).abs().mean().item()
        # print(f"[Debug] Mean abs diff between pred timesteps: {pred_diff:.6f}")

        if visualize_dir is not None:
            os.makedirs(visualize_dir, exist_ok=True)
            for i, (image, track) in enumerate(zip(input_image, pred_tracking)):
                visualize_pred_tracking(
                    init_obs=image.cpu().numpy() * 255,
                    pred_tracking=track.cpu().numpy(),
                    relative_flow=True,
                    denoise_flow=True,
                    max_flow=max_flow, 
                    min_flow=min_flow,
                    save_path=f"{visualize_dir}/ckpt_{epoch}_{i}.gif",
                )
            
            # for i, (image, track) in enumerate(zip(input_image, tracking)):
            #     visualize_pred_tracking(
            #         init_obs=image.cpu().numpy() * 255,
            #         pred_tracking=track.cpu().numpy(),
            #         relative_flow=True,
            #         max_flow=max_flow, 
            #         min_flow=min_flow,
            #         save_path=f"{visualize_dir}/gt_{epoch}_{i}.gif",
            #     )

    eval_loss = sum(eval_loss) / len(eval_loss)
    print(colored("[Eval] ", "green") + f"Epoch {epoch} evaluation loss: {eval_loss:.4f}")
    model.train()
    return eval_loss



@hydra.main(version_base="1.1", config_path="conf", config_name="pred_track_vit")
def main(cfg: DictConfig) -> None:
    # set_seed()
    if cfg.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        # if not check_rank_zero():
        #     cfg.output_dir = os.devnull
    else:
        local_rank = 0
    train(cfg, local_rank=local_rank)

if __name__ == "__main__":
    if dist.is_initialized() and dist.get_rank() != 0: # Just to avoid creating multiple output directories
        # Hijack sys.argv to avoid creating output dir
        sys.argv.append("hydra.run.dir=null")
        sys.argv.append("hydra.output_subdir=null")
    main()

