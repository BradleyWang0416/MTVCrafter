import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from models import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder
from dataset_byBradley import SkeletonDataset
from draw_pose import get_pose_images

import sys
sys.path.append("/home/wxs/Skeleton-in-Context-tpami/")
from lib.utils.viz_skel_seq import viz_skel_seq_anim

print('\npython ' + ' '.join(sys.argv))
"""
CUDA_VISIBLE_DEVICES=6 python -m debugpy --listen 5678 --wait-for-client test_vqvae.py --resume_pth vqvae_experiment/models/checkpoint_epoch_117_step_200000 --load_data_file "/data2/wxs/DATASETS/AMASS_ByBradley/" --data_mode joint3d --num_vis_samples 0
"""

def get_args():
    parser = argparse.ArgumentParser()
    # Model and Data
    parser.add_argument('--resume_pth', type=str, required=True, help="Path to the trained VQVAE model checkpoint.")
    parser.add_argument('--load_data_file', type=str, required=True, help="Path to the dataset file (e.g., a .pkl or .joblib file).")
    parser.add_argument('--data_mode', type=str, default="joint3d", choices=['joint2d', 'joint3d'], help="Specify the data mode for SkeletonDataset.")
    parser.add_argument('--num_frames', type=int, default=49, help="Number of frames per sample.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing.")
    
    # Model Config (should match training)
    parser.add_argument('--nb_code', type=int, default=8192, help="Number of vectors in the codebook.")
    parser.add_argument('--codebook_dim', type=int, default=3072, help="Dimension of each codebook vector.")

    # Output
    parser.add_argument('--output_dir', type=str, default='./test_output', help="Directory to save visualization results.")
    parser.add_argument('--num_vis_samples', type=int, default=5, help="Number of samples to visualize.")

    # Environment
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the test on.")

    return parser.parse_args()

def test_vqvae(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {args.output_dir}")

    # --- 1. Load Model ---
    print("Loading model...")
    encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=args.codebook_dim, downsample_time=[2, 2], downsample_joint=[1, 1])
    vq = VectorQuantizer(nb_code=args.nb_code, code_dim=args.codebook_dim, is_train=False) # Set is_train=False for inference
    decoder = Decoder(in_channels=args.codebook_dim, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    vqvae = SMPL_VQVAE(encoder, decoder, vq)

    # Load checkpoint
    state_dict = {}
    if os.path.isdir(args.resume_pth): # Handle Accelerate saved state directory
        safetensors_path = os.path.join(args.resume_pth, "model.safetensors")
        pytorch_bin_path = os.path.join(args.resume_pth, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            print(f"Loading model from {safetensors_path}")
            state_dict = load_safetensors(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_bin_path):
            print(f"Loading model from {pytorch_bin_path}")
            state_dict = torch.load(pytorch_bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {args.resume_pth}")
    elif os.path.isfile(args.resume_pth): # Handle raw model file
        pass
        # ... (logic for loading a single file)
    else:
        raise FileNotFoundError(f"Checkpoint path not found: {args.resume_pth}")

    # Adjust keys if necessary (e.g., removing 'module.' prefix from DDP)
    unwrapped_state_dict = {}
    is_ddp_model = all(key.startswith('module.') for key in state_dict.keys())
    if is_ddp_model:
        print("Unwrapping model from DDP 'module.' prefix.")
        for k, v in state_dict.items():
            unwrapped_state_dict[k[7:]] = v
        state_dict = unwrapped_state_dict
    
    missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=False)
    print(f"Loaded model with missing keys: {missing_keys}")
    print(f"Loaded model with unexpected keys: {unexpected_keys}")

    vqvae = vqvae.to(args.device)
    vqvae.eval()
    print("Model loaded successfully.")

    # --- 2. Load Dataset ---
    print("Loading dataset...")
    # Note: We assume SkeletonDataset can be modified or used to load a 'test' split.
    # Here, we load the 'train' split as a placeholder for testing purposes.
    # For a real test set, you might need to adjust SkeletonDataset.
    dataset = SkeletonDataset(num_frames=args.num_frames, load_data_file=args.load_data_file, data_mode=args.data_mode, designated_split='test')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for testing
        drop_last=False
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- 3. Run Inference and Evaluation ---
    total_l1_loss = 0.0
    loss_fn = torch.nn.L1Loss()
    vis_count = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            batch = batch.to(args.device)
            
            # Forward pass
            recon_data, loss_commit = vqvae(batch)
            
            # Ensure batch and recon_data have same length for loss calculation
            min_len = min(batch.shape[1], recon_data.shape[1])
            
            # Calculate loss
            reconstruction_loss = loss_fn(recon_data[:, :min_len], batch[:, :min_len])
            total_l1_loss += reconstruction_loss.item()

            # --- 4. Visualization ---
            if vis_count < args.num_vis_samples:
                for j in range(min(args.batch_size, args.num_vis_samples - vis_count)):
                    original_poses = batch[j, :min_len].cpu().numpy()
                    reconstructed_poses = recon_data[j, :min_len].cpu().numpy()

                    # Get pose images (list of PIL Images)
                    # Assuming the data is 3D and can be projected.
                    # You might need to adjust offsets based on your data's coordinate space.
                    if args.data_mode == 'joint3d':
                        original_vis = get_pose_images(original_poses, offset=(512, 512))
                        reconstructed_vis = get_pose_images(reconstructed_poses, offset=(512, 512))

                        # Save a few frames (e.g., first, middle, last)
                        indices_to_save = [0, len(original_vis) // 2, len(original_vis) - 1]
                        for frame_idx in indices_to_save:
                            img_orig = original_vis[frame_idx]
                            img_recon = reconstructed_vis[frame_idx]
                            
                            # Concatenate side-by-side
                            combined_img = Image.new('RGB', (img_orig.width * 2, img_orig.height))
                            combined_img.paste(img_orig, (0, 0))
                            combined_img.paste(img_recon, (img_orig.width, 0))

                            save_path = os.path.join(args.output_dir, f"sample_{vis_count}_frame_{frame_idx}.png")
                            combined_img.save(save_path)
                    
                    vis_count += 1
                    if vis_count >= args.num_vis_samples:
                        break

    avg_l1_loss = total_l1_loss / len(dataloader)
    print(f"\n--- Test Results ---")
    print(f"Average L1 Reconstruction Loss: {avg_l1_loss:.6f}")
    print(f"Visualizations saved in: {args.output_dir}")


if __name__ == '__main__':
    args = get_args()
    test_vqvae(args)
