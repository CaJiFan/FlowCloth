# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffusers import DDPMScheduler
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf
from IPython.display import HTML

from uniclothdiff.registry import build_model, build_dataset
from uniclothdiff.pipelines.cloth_state_est_pipeline import ClothStateEstPipeline

# %%
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "fm"
step = 350000
mode = "full"
config_path = f"./configs/train_state_est_{mode}_{model_type}.yaml"
checkpoint_dir = f"./experiments/vr_{model_type}_{mode}/checkpoints/checkpoint-{step}"

# %%
def load_trained_model(config_path, checkpoint_dir, device):
    """Rebuilds the architecture and loads the trained weights natively."""
    print("Loading configuration...")
    config = OmegaConf.load(config_path)
    
    # 1. Build a dummy model just to get the correct Python Class 
    dummy_model = build_model(OmegaConf.to_container(config.model_cfg))
    ModelClass = type(dummy_model)
    
    # 2. Use Diffusers' native loading method
    print(f"Loading weights from {checkpoint_dir}/model ...")
    model = ModelClass.from_pretrained(checkpoint_dir, subfolder="model")
    model.to(device, dtype=torch.float32)
    model.eval()
    print("Weights loaded successfully!")
    
    # 3. Initialize the Diffusion Pipeline
    diff_dict = OmegaConf.to_container(config.diffusion_cfg)
    if 'type' in diff_dict:
        diff_dict.pop('type')
        
    scheduler = DDPMScheduler(**diff_dict)
    pipeline = ClothStateEstPipeline(model=model, scheduler=scheduler)
    pipeline.to(device, dtype=torch.float32)
    
    return pipeline, config

def run_autoregressive_rollout(pipeline, dataset, seq_idx, num_steps, device):
    """
    Feeds the model's prediction for frame t back in as q_prev for frame t+1.
    """
    print(f"Running {num_steps}-step autoregressive rollout...")
    
    # Grab the sequence data (assuming dataset gives access to full sequences)
    # Adjust this logic based on how your dataset stores valid_pairs/sequences
    xml_file, start_t = dataset.valid_pairs[seq_idx]
    seq_tensor = dataset.sequences[xml_file]
    
    # Setup storage
    predictions = []
    ground_truths = []
    
    # Initialize with the true Frame 0
    current_q_prev = seq_tensor[start_t - 1].unsqueeze(0).to(device, dtype=torch.float32)
    predictions.append(current_q_prev.squeeze(0).cpu().numpy())
    ground_truths.append(current_q_prev.squeeze(0).cpu().numpy())
    
    with torch.no_grad():
        for t in range(start_t, start_t + num_steps):
            # 1. Grab Ground Truths for this step
            q_gt = seq_tensor[t].unsqueeze(0).to(device, dtype=torch.float32)
            pcd = dataset._simulate_depth_camera(seq_tensor[t]).unsqueeze(0).to(device, dtype=torch.float32)
            
            # (Optional) Grab other inputs your pipeline might need like q_temp or action
            
            # 2. The Pipeline Magic! 
            # We feed it our CURRENT_Q_PREV (which is the model's prediction from the last step)
            # You might need to adjust kwargs based on your exact ClothStateEstPipeline signature
            output = pipeline(
                q_prev=current_q_prev,
                pcd=pcd,
                num_inference_steps=50, # Fast DDPM inference
                output_type="tensor"
            )
            
            pred_q_gt = output.tensor if hasattr(output, 'tensor') else output[0]
            
            # 3. Store and Roll Forward
            predictions.append(pred_q_gt.squeeze(0).cpu().numpy())
            ground_truths.append(q_gt.squeeze(0).cpu().numpy())
            
            # Autoregressive update! The prediction becomes the next step's input.
            current_q_prev = pred_q_gt
            
    return np.array(predictions), np.array(ground_truths)

def animate_rollout(predictions, ground_truths, save_path="rollout.gif"):
    """Creates a side-by-side 3D animation of the prediction vs ground truth."""
    fig = plt.figure(figsize=(12, 6))
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_gt = fig.add_subplot(122, projection='3d')
    
    def update(frame_idx):
        ax_pred.clear()
        ax_gt.clear()
        
        pred = predictions[frame_idx]
        gt = ground_truths[frame_idx]
        
        ax_pred.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', s=20)
        ax_gt.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', s=20)
        
        ax_pred.set_title(f"Prediction (Step {frame_idx})")
        ax_gt.set_title(f"Ground Truth (Step {frame_idx})")
        
        for ax in [ax_pred, ax_gt]:
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([0, 0.6])
            ax.view_init(elev=30, azim=45)
            
    anim = FuncAnimation(fig, update, frames=len(predictions), interval=200)
    anim.save(save_path, dpi=80, writer='pillow')
    print(f"Animation saved to {save_path}!")
    plt.close()

def run_state_estimation_sequence(pipeline, dataset, seq_idx, num_steps, device):
    """
    Estimates the cloth state frame-by-frame using the camera point cloud and template.
    Reverses the mean-centering to plot in absolute world coordinates.
    """
    print(f"Running {num_steps}-step State Estimation...")
    
    # Grab the sequence data
    xml_file, start_t = dataset.valid_pairs[seq_idx]
    seq_tensor = dataset.sequences[xml_file]
    
    predictions_world = []
    ground_truths_world = []
    
    with torch.no_grad():
        for t in range(start_t, start_t + num_steps):
            
            # 1. Replicate Dataset Preprocessing Manually for continuous flow
            q_prev = seq_tensor[t - 1].clone()
            q_gt = seq_tensor[t].clone()
            
            # Mean-Centering (Exactly as your dataset does it)
            center_of_mass = q_prev.mean(dim=0)
            q_gt_centered = q_gt - center_of_mass
            q_temp_centered = seq_tensor[0].clone() - center_of_mass
            
            # Simulate camera on the current GT frame
            pcd = dataset._simulate_depth_camera(q_gt)
            
            # 2. Slice for Edge-Only model (if applicable)
            if getattr(dataset, 'train_mode', 'full') == "edge_only":
                q_gt_centered = q_gt_centered[dataset.contour_idx]
                q_temp_centered = q_temp_centered[dataset.contour_idx]
                
            # Move to device and add batch dimension
            pcd = pcd.unsqueeze(0).to(device, dtype=torch.float32)
            q_temp_centered = q_temp_centered.unsqueeze(0).to(device, dtype=torch.float32)
            
            num_points = q_temp_centered.shape[1]
            noise_shape = (1, num_points, 3)
            

            # 3. The Pipeline Magic! (Matching your validation script exactly)
            output = pipeline(
                encoder_hidden_states=pcd,
                q_temp=q_temp_centered,
                shape=noise_shape,
                num_inference_steps=50,  # Denoise 20x faster for viz!
                call_v2=True,
                do_classifier_free_guidance=False # Usually False for pure state-est
            )
            
            pred_centered = output.frames if hasattr(output, 'frames') else output[0]
            if isinstance(pred_centered, torch.Tensor):
                pred_centered = pred_centered.squeeze(0).cpu().numpy()
                
            # 4. Undo the Mean-Centering to get World Coordinates
            com_np = center_of_mass.cpu().numpy()
            pred_world = pred_centered + com_np
            gt_world = q_gt_centered.cpu().numpy() + com_np
            
            predictions_world.append(pred_world)
            ground_truths_world.append(gt_world)
            
    return np.array(predictions_world), np.array(ground_truths_world)

def animate_comparison(predictions, ground_truths, save_path="state_est_result.gif"):
    """Creates a side-by-side 3D animation of the prediction vs ground truth."""
    fig = plt.figure(figsize=(12, 6))
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_gt = fig.add_subplot(122, projection='3d')
    
    def update(frame_idx):
        ax_pred.clear()
        ax_gt.clear()
        
        pred = predictions[frame_idx]
        gt = ground_truths[frame_idx]
        
        # Plot Prediction (Red) and GT (Blue)
        ax_pred.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', s=30, alpha=0.8)
        ax_gt.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', s=30, alpha=0.8)
        
        ax_pred.set_title(f"State Est Prediction (Frame {frame_idx})")
        ax_gt.set_title(f"Ground Truth (Frame {frame_idx})")
        
        for ax in [ax_pred, ax_gt]:
            # Adjust these limits based on your VR workspace scale (e.g., meters)
            ax.set_xlim([-0.8, 0.8])
            ax.set_ylim([-0.8, 0.8])
            ax.set_zlim([0, 1.0])
            ax.view_init(elev=30, azim=45)
            
    anim = FuncAnimation(fig, update, frames=len(predictions), interval=200)
    anim.save(save_path, dpi=80, writer='pillow')
    print(f"Animation saved to {save_path}!")
    plt.close()



        
# 1. Load your Pipeline
pipe, cfg = load_trained_model(
    config_path=config_path,
    checkpoint_dir=checkpoint_dir,
    device=device
)

# 2. Load Validation Dataset
val_dataset_cfg = OmegaConf.to_container(cfg.dataset_cfg)
val_dataset_cfg["mode"] = "val"
val_dataset_cfg["train_mode"] = mode # Or "full" depending on the model
val_dataset = build_dataset(val_dataset_cfg) # Make sure it loads Val data

preds, gts = run_state_estimation_sequence(
    pipeline=pipe, 
    dataset=val_dataset, 
    seq_idx=50,       # Pick a random sequence to test
    num_steps=30,     # Render 30 frames
    device=device
)

# 4. Create the GIF!
animate_comparison(preds, gts, save_path=f"vr_state_est_{model_type}_{mode}_{step}.gif")
        



