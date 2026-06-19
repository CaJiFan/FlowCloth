import os
import math
import shutil
import logging
import importlib
import numpy as np

import torch
from torch.utils.data import RandomSampler
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import random
import numpy as _np
import os as _os
from accelerate import DistributedDataParallelKwargs

import diffusers
from diffusers.optimization import get_scheduler
import wandb
from args import parse_args
from uniclothdiff.registry import (
    build_dataset, 
    build_model, 
    build_scheduler
)
from uniclothdiff.utils.torch_utils import to_torch_dtype
from uniclothdiff.utils.training_utils import (
    get_model_numel, 
    format_numel_str, 
    backup_code, 
    find_unused_parameters,
    get_model_parameters,
    corner_localization_error,
    chamfer_l1,
    f_score
)
from uniclothdiff.pipelines.cloth_dynamics_pipeline import ClothDynamicsPipeline
from uniclothdiff.pipelines.cloth_state_est_pipeline import ClothStateEstPipeline
from uniclothdiff.pipelines.cloth_state_est_fm_pipeline import ClothStateEstFMPipeline
from tqdm.auto import tqdm
from tqdm import tqdm
from omegaconf import OmegaConf


logger = get_logger(__name__, log_level="INFO")

def setup_accelerator(project_dir: str, 
                      logging_dir: str, 
                      gradient_accumulation_steps: int,
                      mixed_precision: str,
                      logs_report_to: str,
                      ):
    accelerator_project_config = ProjectConfiguration(
        project_dir=project_dir, logging_dir=logging_dir)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logs_report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    
    return accelerator

def main():
    assert torch.cuda.is_available(), \
        "Training requires at least one GPU."
        
    args = parse_args()
    args_dict = vars(args)
    exp_cfg = OmegaConf.load(args.config)
    print('exp_cfg:', exp_cfg)
    config = OmegaConf.create(args_dict)
    print('config:', config)
    config = OmegaConf.merge(config, exp_cfg)
    # config.dataset_cfg.data_dir = args.data_dir 
    # Handle the repository creation
    experiment_dir = os.path.join("experiments", f"{config.exp_name}")
    acc_logging_dir = os.path.join(experiment_dir, config.logging_dir)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    
    accelerator = setup_accelerator(
        experiment_dir, 
        acc_logging_dir, 
        config.gradient_accumulation_steps,
        config.mixed_precision,
        config.report_to
    )
    
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
    
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    if accelerator.is_main_process and config.exp_name != "DEBUG":
        # Allow disabling code backup (to save space) via env var
        # if os.getenv("DISABLE_BACKUP_CODE", "0") != "1":
        #     backup_code(experiment_dir, logger=logger)
        OmegaConf.save(config, os.path.join(experiment_dir, "config.yml"))
        # wandb.init(
        #     project=config.wandb_cfg.project_name,
        #     entity=config.wandb_cfg.entity,
        #     tags=config.wandb_cfg.tags,
        #     name=config.exp_name
        # )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    def seed_everything(seed: int):
        # Robust seeding across python/numpy/torch and accelerate
        try:
            _os.environ['PYTHONHASHSEED'] = str(seed)
        except Exception:
            pass
        random.seed(seed)
        _np.random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
            # Make deterministic behaviour a user choice; we enable conservative settings here
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            set_seed(seed)
        except Exception:
            pass

    seed_everything(int(config.seed))
    
    diffusion_scheduler = build_scheduler(OmegaConf.to_container(config.diffusion_cfg))
    
    if config.pretrained_model_name_or_path:
        model_cls = getattr(importlib.import_module("uniclothdiff.models"), config.model_cfg.type)
        model = model_cls.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="model",
            low_cpu_mem_usage=False,
        )
    else:
        model = build_model(OmegaConf.to_container(config.model_cfg))
        
    model_numel, model_numel_trainable = get_model_numel(model)
    # update_params_name, update_params= get_model_parameters(model)        # uncomment it to find unused params
    
    cfg_dtype = config.get("mixed_precision", "float32")
    weight_dtype = to_torch_dtype(cfg_dtype)

    model.requires_grad_(False)
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "model"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            
            class_name = type(model).__name__  # get object class name
            module_name = model.__module__  # get the module where class is in
            module = importlib.import_module(module_name)  # import module
            ModelClass = getattr(module, class_name)  # import class
            
            # load diffusers style into model
            load_model = ModelClass.from_pretrained(input_dir, subfolder="model")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    print('using scaled learning rate?', config.scale_lr)
    print('learning rate before accelerator preparation:', config.learning_rate)
    
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing() 
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps *
            config.per_gpu_batch_size * accelerator.num_processes
        )

    print('learning rate after accelerator preparation:', config.learning_rate)
    print(f" Sampling steps = {config.sampling_steps}")
    
    optimizer_cls = torch.optim.AdamW
    model.requires_grad_(True)

    optimizer = optimizer_cls(
        model.parameters(),
        # update_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    
    # DataLoaders creation:
    config.global_batch_size = config.per_gpu_batch_size * accelerator.num_processes
    
    train_dataset_cfg = OmegaConf.to_container(config.dataset_cfg)
    train_dataset_cfg["mode"] = "train"
    valid_dataset_cfg = OmegaConf.to_container(config.dataset_cfg)
    valid_dataset_cfg["mode"] = "val"
    train_dataset = build_dataset(train_dataset_cfg)
    valid_dataset = build_dataset(valid_dataset_cfg)
    
    data_sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=data_sampler,
        batch_size=config.per_gpu_batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.per_gpu_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True
    )

    print('max train steps before accelerator preparation:', config.max_train_steps)
    
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader
    )

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model._set_static_graph()
    
    # length of dataloader may change after prepared with accelerator
    num_update_step_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_step_per_epoch
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_step_per_epoch)

    config.do_classifier_free_guidance = False
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ClothDiffusion", 
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": config.exp_name}}
        )

    # Train!
    total_batch_size = config.per_gpu_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    
    logger.info("******** Running training ********")
    logger.info(f'  Train mode = {train_dataset_cfg.get("train_mode", "full")}')
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info("  Total model prarameter = %s", format_numel_str(model_numel),)
    logger.info("  Trainable model prarameter = %s", format_numel_str(model_numel_trainable),)
    logger.info(f"  Do classifier free guidance = {config.do_classifier_free_guidance}")
    logger.info(f"  LR Scheduler = {config.lr_scheduler}")
    logger.info(f" Sampling steps = {config.sampling_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            # path = os.path.basename(args.resume_from_checkpoint)
            path = config.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(checkpoints_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Loading state from checkpoint {path}")
            if os.path.exists(path):
                accelerator.print(f"Fine-tuning with previous experiments checkpoint")
                accelerator.load_state(path)
                global_step = 0
            else:
                accelerator.print(f"Resuming from previous experiments")
                accelerator.load_state(os.path.join(checkpoints_dir, path))
                global_step = int(path.split("-")[1])

            resume_global_step = global_step * config.gradient_accumulation_steps
            first_epoch = global_step // num_update_step_per_epoch
            resume_step = resume_global_step % (num_update_step_per_epoch * config.gradient_accumulation_steps)

    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    
    # logger.info("GRABBING SINGLE BATCH FOR OVERFIT TEST...")
    # sanity_batch = next(iter(train_dataloader))


    stop_training = False
    for epoch in range(first_epoch, config.num_train_epochs):
        if stop_training:
            break
        model.train()
        train_loss = 0.0

        # ******** training loop in an epoch ******** #
        for step, batch in enumerate(train_dataloader):
            # batch = sanity_batch.copy() 
            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % config.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                # input = batch.pop('q_delta')        # this is for training dynamics
                if 'q_gt' in batch:
                    input = batch.pop('q_gt')       # State Estimation uses the Ground Truth Mesh
                else:
                    input = batch.pop('q_delta')

                if "StateEst" in config.model_cfg.type:
                    # diffusion_scheduler is already the correct class (DDPM or FM) built
                    # from diffusion_cfg.type — both implement training_losses_with_cfg
                    loss = diffusion_scheduler.training_losses_with_cfg(
                        model=model, 
                        input=input,
                        model_kwargs=batch,
                        weight_dtype=weight_dtype,
                        generator=generator
                    )
                else:
                    loss = diffusion_scheduler.training_losses(
                        model=model, 
                        input=input,
                        model_kwargs=batch,
                        weight_dtype=weight_dtype
                    )

                avg_loss = accelerator.gather(loss.repeat(config.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients and config.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    accelerator.log({"train/grad_norm": grad_norm.item()}, step=global_step)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # accelerator.print(f"Step {global_step:05d}, Epoch-Step {epoch:03d}-{step:03d}, loss: {train_loss}") 
                current_lr = optimizer.param_groups[0]['lr']

                # Update the progress bar dynamically without breaking to a new line
                pbar_updates = {
                    "epoch": f"{epoch:03d}",
                    "global_step": global_step,
                    "loss": f"{train_loss:.4f}",
                    "lr": f"{current_lr:.2e}"
                }
                progress_bar.set_postfix(pbar_updates)

                accelerator.log({"train/loss": train_loss}, step=global_step)
                accelerator.log({"train/learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
                # Log per-component losses from the diffusion scheduler if available
                comp_logs = {}
                if hasattr(diffusion_scheduler, "last_loss_noise") and diffusion_scheduler.last_loss_noise is not None:
                    comp_logs["train/loss_noise"] = diffusion_scheduler.last_loss_noise
                if hasattr(diffusion_scheduler, "last_loss_shape") and diffusion_scheduler.last_loss_shape is not None:
                    comp_logs["train/loss_shape"] = diffusion_scheduler.last_loss_shape
                if len(comp_logs) > 0:
                    accelerator.log(comp_logs, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint
                if accelerator.is_main_process and \
                    global_step % config.checkpointing_steps == 0:
                    logger.info("Saving checkpoint")
                    if not os.path.exists(checkpoints_dir):
                        os.makedirs(checkpoints_dir)
                    checkpoints = os.listdir(checkpoints_dir)
                    checkpoints = [
                        d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(
                        checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= config.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(
                            f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(
                                checkpoints_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        checkpoints_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                # Run sampling
                if accelerator.is_main_process and \
                    (global_step % config.sampling_steps == 0 or \
                     global_step == 1):
                    logger.info("***** Running sampling *****")
                    
                    if "StateEst" in config.model_cfg.type:
                        use_flow_matching = "FlowMatching" in config.diffusion_cfg.type
                        if use_flow_matching:
                            pipeline = ClothStateEstFMPipeline(
                                model=accelerator.unwrap_model(model),
                                scheduler=accelerator.unwrap_model(diffusion_scheduler)
                            )
                        else:
                            pipeline = ClothStateEstPipeline(
                                model=accelerator.unwrap_model(model),
                                scheduler=accelerator.unwrap_model(diffusion_scheduler)
                            )
                        is_dynamics = False
                    else:
                        from uniclothdiff.pipelines import ClothDynamicsPipeline
                        pipeline = ClothDynamicsPipeline(
                            model=accelerator.unwrap_model(model),
                            scheduler=accelerator.unwrap_model(diffusion_scheduler)
                        )
                        is_dynamics = True

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # corner indices depend on both mode (edge/full) and grid size (dataset-specific)
                    is_edge_mode = train_dataset_cfg.get("train_mode", "full") == "edge_only"
                    _ds_type = train_dataset_cfg.get("type", "ClothTrackingDataset")
                    if "TRTM" in _ds_type:
                        # 21×21 grid: full corners [0,20,420,440], edge contour corners [0,20,59,79]
                        val_corner_indices = [0, 20, 59, 79] if is_edge_mode else [0, 20, 420, 440]
                    else:
                        # 20×20 grid (VR-Folding): full corners [0,19,380,399], edge [0,19,56,75]
                        val_corner_indices = [0, 19, 56, 75] if is_edge_mode else [0, 19, 380, 399]

                    with accelerator.autocast():
                        sample_valid_loss = 0.0
                        sample_corner_error = 0.0
                        sample_chamfer_error = 0.0
                        sample_fscore = 0.0
                        num_sample_batches = 0

                        for val_step, val_batch in enumerate(valid_dataloader):
                            if val_step >= 5: # evaluate up to 5 batches for speed
                                break
                            # val_batch = sanity_batch.copy()
                            batch_size = val_batch['pcd'].shape[0] if not is_dynamics else val_batch['q_prev'].shape[0]

                            if is_dynamics:
                                # --- ORIGINAL DYNAMICS LOGIC ---
                                q_prev = val_batch['q_prev'].to(accelerator.device)
                                action = val_batch['action'].to(accelerator.device)
                                q_mask = val_batch['mask'].to(accelerator.device)
                                
                                pred = pipeline(
                                    q_prev=q_prev,
                                    q_mask=q_mask,
                                    action=action,
                                    do_classifier_free_guidance=config.do_classifier_free_guidance
                                )[0]
                                
                                if q_prev.ndim == 5:
                                    q_next = val_batch['q_next'].reshape(batch_size, val_batch['action'].shape[1], 3, -1).permute(0, 1, 3, 2).cpu().numpy()
                                elif q_prev.ndim == 4:
                                    q_next = val_batch['q_next'].cpu().numpy()
                                    
                                target = q_next

                            else:
                                # --- NEW STATE ESTIMATION LOGIC (TRTM) ---
                                pcd = val_batch['pcd'].to(accelerator.device)
                                q_temp = val_batch['q_temp'].to(accelerator.device)
                                q_gt = val_batch['q_gt'].to(accelerator.device)
                                num_nodes = q_temp.shape[1] 
                                noise_shape = (batch_size, num_nodes, 3)

                                # FM needs only 50 steps; DDPM uses 100 as a fast approximation
                                _use_fm = "FlowMatching" in config.diffusion_cfg.type
                                pred_output = pipeline(
                                    encoder_hidden_states=pcd,
                                    q_temp=q_temp,
                                    shape=noise_shape,
                                    num_inference_steps=50 if _use_fm else 100,
                                    do_classifier_free_guidance=config.do_classifier_free_guidance,
                                    call_v2=True,
                                )

                                pred = pred_output.frames
                                # If pred is a tensor, convert to numpy for MSE
                                if isinstance(pred, torch.Tensor):
                                    pred = pred.cpu().numpy()
                                target = q_gt.cpu().numpy()

                                if val_step == 0: 
                                    # Shape must be [num_points, 3]
                                    # print(f"pred shape: {pred.shape}, target shape: {target.shape}")
                                    pred_pc = pred[0]  
                                    gt_pc = target[0]  
                                    
                                    accelerator.log({
                                        "val/point_cloud_pred": wandb.Object3D(pred_pc),
                                        "val/point_cloud_gt": wandb.Object3D(gt_pc),
                                        # "val/camera_input_pcd": wandb.Object3D(pcd[0].cpu().numpy())
                                    }, step=global_step)

                                    # spatial_spread = pred.std(dim=1).mean().item()
                                    spatial_spread = pred.std(axis=1).mean().item()
                                    accelerator.log({"val/spatial_spread": spatial_spread}, step=global_step)
                                
                            # Calculate validation error
                            squared_error = (pred - target) ** 2
                            mse_error = np.mean(squared_error.reshape(batch_size, -1), axis=1).mean()
                            corner_error = corner_localization_error(pred, target, corner_indices=val_corner_indices)
                            chamfer_error = chamfer_l1(pred, target)
                            fscore = f_score(pred, target)
                            
                            sample_valid_loss += mse_error
                            sample_corner_error += corner_error
                            sample_chamfer_error += chamfer_error
                            sample_fscore += fscore

                            num_sample_batches += 1
                            
                    sample_valid_loss /= num_sample_batches
                    sample_corner_error /= num_sample_batches
                    sample_chamfer_error /= num_sample_batches
                    sample_fscore /= num_sample_batches
                    # accelerator.print(f"Validation Loss (sampling): {sample_valid_loss}")
                    accelerator.log({"val/sampled_valid_loss": sample_valid_loss}, step=global_step)
                    accelerator.log({"val/sampled_corner_error": sample_corner_error}, step=global_step)
                    accelerator.log({"val/sampled_chamfer_error": sample_chamfer_error}, step=global_step)
                    accelerator.log({"val/sampled_fscore": sample_fscore}, step=global_step)

                    pbar_updates['sampled_val_loss'] = f"{sample_valid_loss:.4f}"
                    progress_bar.set_postfix(pbar_updates)

                    # avg_train_loss = train_loss / config.sampling_steps
                    # overfit_gap = sample_valid_loss - avg_train_loss
                    # accelerator.log({"val/overfit_gap": overfit_gap}, step=global_step)

                    # train_loss = 0.0

                    del pipeline
                    torch.cuda.empty_cache()

                    if global_step >= 251000:
                        print("Max step reached, ending training...")
                        stop_training = True
                        break
                
                # if global_step % config.sampling_steps == 0 or global_step == 1:
                #     accelerator.wait_for_everyone()
                    

    accelerator.wait_for_everyone()
    accelerator.end_training()
if __name__ == "__main__":
    main()




# python scripts/train.py \
#   --data_dir=/home/cjimenez/Projects/Tracking/Code/TrackAnyCloth/TRTM/datasets/template_square \
#   --per_gpu_batch_size=256 \
#   --num_train_epochs=500 \
#   --checkpointing_steps=2000 \
#   --checkpoints_total_limit=10 \
#   --gradient_accumulation_steps=8 \
#   --learning_rate=1e-4 \
#   --lr_warmup_steps=500 \
#   --lr_scheduler="cosine" \
#   --seed=1 \
#   --num_workers=12 \
#   --mixed_precision="fp16" \
#   --exp_name="vr_tracking_mode" \
#   --resume_from_checkpoint="latest" \
#   --pretrained_model_name_or_path="None" \
#   --config="configs/train_state_est_full.yaml" 