# Standard Library
import logging
from pathlib import Path

# External Modules
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Internal Modules
from kmspeaker.audio.audio_factory import get_audio_model
from kmspeaker.data.data_factory import get_data
from kmspeaker.models.checkpoint_manager import CheckpointManager
from kmspeaker.models.factory import get_model
from kmspeaker.training.flow_trainer import FlowTrainer
from kmspeaker.utils.train import (
    initialize_optional_models, 
    resume_from_checkpoint, 
    log_model_parameter_counts, 
    build_optimizer, 
    build_lr_scheduler, 
    build_checkpoint_data
)

# Setup logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="train_km_speaker", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting training script...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = cfg.trainer.device

    data_root = Path(cfg.data_root)

    base_logdir = Path(cfg.base_logdir)

    ### Initialize dataset and dataloader
    logger.info("Initializing dataset and dataloader...")
    dataset = get_data(cfg.data, device)

    pca_decode_metadata = None
    if cfg.data.decode_pca_in_dataset and cfg.data.target_codes == "pca":
        pca_decode_metadata = dataset.get_pca_decode_metadata()

    train_sampler = RandomSampler(dataset, replacement=True, num_samples=cfg.data.batch_size * cfg.trainer.total_training_steps)
    train_loader = DataLoader(dataset, batch_size=cfg.data.batch_size, sampler=train_sampler, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
    logger.info(f"Dataset size: {len(dataset)}, Batch size: {cfg.data.batch_size}, DataLoader steps per epoch: {len(train_loader)}")

    logger.info("Initializing audio encoder...")
    audio_encoder = get_audio_model(cfg.audio_encoder).to(device)

    ################### Model Initialization ###################
    logger.info("Initializing model...")
    model = get_model(cfg.model, audio_encoder, num_keypoint=dataset.num_keypoint)
    model.to(device)

    logger.info("Initializing optional models:")
    keypoint_style_encoder = initialize_optional_models(
        cfg=cfg,
        device=device,
        num_keypoint=dataset.num_keypoint,
        num_extra_style_keypoint=dataset.num_extra_style_keypoint,
    )

    ############ Resume from checkpoint if specified ############
    resume_from_checkpoint(cfg=cfg, base_logdir=base_logdir, device=device, model=model, keypoint_style_encoder=keypoint_style_encoder)

    ### Log model parameter counts
    log_model_parameter_counts(
        model=model,
        keypoint_style_encoder=keypoint_style_encoder,
    )

    ### Build optimizer and LR scheduler
    optimizer = build_optimizer(
        cfg=cfg,
        model=model,
    )

    # Initialize LR scheduler if specified
    lr_scheduler = build_lr_scheduler(cfg=cfg, optimizer=optimizer)

    # Create checkpoint directory if it doesn't exist
    logdir = Path(cfg.logdir)
    checkpoint_dir = logdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard summary writer
    tensorboard_dir = logdir / f"tb_{cfg.exp_name}" if cfg.exp_name is not None and cfg.exp_name != "" else logdir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=cfg.trainer.max_checkpoints,
        milestone_interval=cfg.trainer.get("milestone_interval", 100000)
    )

    # --- Flow Trainer ---
    logger.info("Initializing FlowTrainer...")
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        training_target=cfg.training_target,
        loss_weights=cfg.loss_weights,
        clip_grad_norm=cfg.trainer.clip_grad_norm if hasattr(cfg.trainer, "clip_grad_norm") else None,
        use_mesh_loss=cfg.trainer.use_mesh_loss,
        pca_decode_metadata=pca_decode_metadata,
        cfg_audio_keypoint_disentanglement=cfg.audio_keypoint_disentanglement if cfg.trainer.audio_keypoint_disentanglement else None,
        data_root=data_root,
        cfg_probability=cfg.trainer.cfg_probability,
        keypoint_style_encoder=keypoint_style_encoder,
        target_codes=cfg.trainer.target_codes,
    )

    # Add keypoint style encoder parameters to optimizer if used (skip if frozen)
    if trainer.use_keypoint_style_encoder and any(p.requires_grad for p in keypoint_style_encoder.parameters()):
        optimizer.add_param_group({"params": keypoint_style_encoder.parameters(), "lr": cfg.optimizer.lr, "weight_decay": cfg.optimizer.weight_decay})
        logger.info("Added keypoint_style_encoder parameters to optimizer.")

    # --- Training Loop ---
    logger.info(f"Starting training for {cfg.trainer.total_training_steps} steps...")
    global_step = 0
    pbar = tqdm(total=cfg.trainer.total_training_steps, desc="Training Steps", disable=not cfg.trainer.use_tqdm)
    while global_step < cfg.trainer.total_training_steps:
        for _, train_batch in enumerate(train_loader):
            loss, loss_dict = trainer.train_step(train_batch)

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

            if global_step % cfg.trainer.log_iter == 0:  # log every n steps
                for key, value in loss_dict.items():
                    summary_writer.add_scalar(f"train/{key}", value, global_step)
                summary_writer.add_scalar("train/total_loss", loss, global_step)
                if lr_scheduler:
                    summary_writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)

            if global_step % cfg.trainer.save_iter == 0:  # save every n steps
                checkpoint_data = build_checkpoint_data(
                    cfg=cfg,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    keypoint_metadata=dataset.get_keypoint_metadata(),
                    keypoint_style_encoder=keypoint_style_encoder,
                )
                checkpoint_manager.save_checkpoint(global_step=global_step, data=checkpoint_data)

            if global_step >= cfg.trainer.total_training_steps:  # break if we reach the max steps
                break
    pbar.close()
    summary_writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
