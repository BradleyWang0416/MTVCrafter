import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file as load_safetensors
import yaml
from easydict import EasyDict as edict
import ast

from config.vision_backbone import config as vision_config
from config.vqvae import vqvae_config
from models import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder
from dataset_byBradley import SkeletonDataset
from draw_pose import get_pose_images
from models import HYBRID_VQVAE

import sys
sys.path.append("../Skeleton-in-Context-tpami/")
from lib.utils.viz_skel_seq import viz_skel_seq_anim
sys.path.append("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset

print('\npython ' + ' '.join(sys.argv))

from accelerate import Accelerator

def update_dict(v, cfg):
    for kk, vv in v.items():
        if kk in cfg:
            if isinstance(vv, dict) and isinstance(cfg[kk], dict):
                update_dict(vv, cfg[kk])
            else:
                if vv is not None:
                    cfg[kk] = vv
        else:
            if vv is not None: 
                cfg[kk] = vv


def update_config(path, args):
    with open(path) as fin: # path = 'experiments/human36m/human36m.yaml'
        exp_config = edict(yaml.safe_load(fin))
        update_dict(vars(args), exp_config)
        return exp_config
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/old_wo_img.yaml', help="Path to config file")
    # Model and Data
    parser.add_argument('--resume_pth', type=str, required=True, help="Path to the trained VQVAE model checkpoint.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing.")
    
    # Model Config (should match training)
    parser.add_argument('--nb_code', type=int, default=8192, help="Number of vectors in the codebook.")
    parser.add_argument('--codebook_dim', type=int, default=3072, help="Dimension of each codebook vector.")

    # Output
    # parser.add_argument('--output_dir', type=str, default='./test_output', help="Directory to save visualization results.")
    # parser.add_argument('--num_vis_samples', type=int, default=5, help="Number of samples to visualize.")

    # Environment
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the test on.")

    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mpjpe', 'mpjpe_millimeter'])
    parser.add_argument('--vqvae_type', type=str, default='base')
    parser.add_argument('--joint_data_type', type=str, default='joint3d_image_affined_normed')

    # Also defined in yaml. 如果在命令行中指定，则覆盖yaml中的配置
    parser.add_argument('--num_frames', type=int, default=None, help="Number of frames per sample.")
    parser.add_argument('--sample_stride', type=int, default=None)
    parser.add_argument('--data_stride', type=int, default=None)
    parser.add_argument('--data_mode', type=str, default=None)

    parser.add_argument('--load_data_file', type=str, default=None)
    parser.add_argument('--load_image_source_file', type=str, default=None)
    parser.add_argument('--load_bbox_file', type=str, default=None)
    parser.add_argument('--load_text_source_file', type=str, default=None)
    parser.add_argument('--return_extra', type=str, default=None)

    parser.add_argument('--normalize', type=str, default=None)
    parser.add_argument('--filter_invalid_images', type=bool, default=None)
    parser.add_argument('--processed_image_shape', type=str, default=None)
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--get_item_list', type=str, default=None)

    # VISION BACKBONE config. 如果在命令行中指定，则覆盖vision_config中的配置
    parser.add_argument('--hrnet_output_level', type=str, default=None, help="int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征")
    parser.add_argument('--vision_guidance_ratio', type=float, default=None)
    
    parser.add_argument('--downsample_time', type=str, default=None)
    parser.add_argument('--frame_upsample_rate', type=str, default=None)

    parser.add_argument('--vision_guidance_where', type=str, default=None)
    parser.add_argument('--vision_guidance_fuse', type=str, default=None)

    args = parser.parse_args()

    if isinstance(args.return_extra, str):
        args.return_extra = ast.literal_eval(args.return_extra)
    if isinstance(args.processed_image_shape, str):
        args.processed_image_shape = ast.literal_eval(args.processed_image_shape)
    if isinstance(args.get_item_list, str):
        args.get_item_list = ast.literal_eval(args.get_item_list)
    if isinstance(args.hrnet_output_level, str):
        args.hrnet_output_level = ast.literal_eval(args.hrnet_output_level)
    if isinstance(args.downsample_time, str):
        args.downsample_time = ast.literal_eval(args.downsample_time)
    if isinstance(args.frame_upsample_rate, str):
        args.frame_upsample_rate = ast.literal_eval(args.frame_upsample_rate)

    config = update_config(args.config, args)

    return config

def test_vqvae(args):
    accelerator = Accelerator()
    device = accelerator.device
    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    # print(f"Visualizations will be saved to: {args.output_dir}")

    # --- 1. Load Model ---
    print("Loading model...")
    vqvae_config.encoder.out_channels = args.codebook_dim
    vqvae_config.vq.nb_code = args.nb_code
    vqvae_config.vq.code_dim = args.codebook_dim
    vqvae_config.decoder.in_channels = args.codebook_dim

    if args.get('downsample_time', None) is not None:
        vqvae_config.encoder.downsample_time = args.downsample_time
    if args.get('frame_upsample_rate', None) is not None:
        vqvae_config.decoder.frame_upsample_rate = args.frame_upsample_rate
    
    if args.get('backbone', None) is not None:
        vision_config.model.backbone.type = args.backbone
    if args.get('hrnet_output_level', None) is not None:
        vision_config.model.hybrid.hrnet_output_level = args.hrnet_output_level
    if args.get('vision_guidance_ratio', None) is not None:
        vision_config.model.hybrid.vision_guidance_ratio = args.vision_guidance_ratio
    if args.get('vision_guidance_where', None) is not None:
        vision_config.model.hybrid.vision_guidance_where = args.vision_guidance_where
    if args.get('vision_guidance_fuse', None) is not None:
        vision_config.model.hybrid.vision_guidance_fuse = args.vision_guidance_fuse
    vqvae_config.vq.is_train = False
    vqvae = HYBRID_VQVAE(vqvae_config.encoder, vqvae_config.decoder, vqvae_config.vq, vision_config=vision_config, joint_data_type=args.joint_data_type)
    if vision_config.model.hybrid.vision_guidance_ratio > 0:
        for p in vqvae.vision_backbone.parameters():
            p.requires_grad = False

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
    # dataset = SkeletonDataset(num_frames=args.num_frames, sample_stride=args.sample_stride, load_data_file=args.load_data_file, data_mode=args.data_mode, designated_split='test')
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False, # No need to shuffle for testing
    #     drop_last=False
    # )
    dataset = Multimodal_Mocap_Dataset( num_frames=args.num_frames, sample_stride=args.sample_stride, data_stride=args.data_stride,
                                        data_mode=args.data_mode,
                                        designated_split='test',
                                        load_data_file=args.load_data_file,
                                        load_image_source_file=args.load_image_source_file,
                                        load_bbox_file=args.load_bbox_file,
                                        load_text_source_file=args.load_text_source_file,
                                        return_extra=args.return_extra,
                                        # data preprocessing config
                                        normalize=args.normalize,  # isotropic (i.e., screen_coordinate_normalize), anisotropic
                                        # image config
                                        filter_invalid_images=args.filter_invalid_images,
                                        processed_image_shape=args.processed_image_shape,    # e.g., (192,256)
                                        backbone=args.backbone,
                                        # dataloader config
                                        get_item_list=args.get_item_list,
                                        batch_return_type='tuple',
                                        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for testing
        drop_last=False,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )

    # 用 accelerator.prepare 包装模型和 dataloader
    vqvae, dataloader = accelerator.prepare(vqvae, dataloader)


    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- 3. Run Inference and Evaluation ---
    total_l1_loss = 0.0
    

    if args.loss_type == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif args.loss_type in ['mpjpe', 'mpjpe_millimeter']:
        def mpjpe_loss(pred, target):
            return torch.mean(torch.norm(pred - target, dim=-1))
        loss_fn = mpjpe_loss

    vis_count = 0
    codebook_usage = torch.zeros(args.nb_code, dtype=torch.long, device='cuda')

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):

            if isinstance(batch, tuple):
                batch_dict = {}
                assert len(batch) == len(dataloader.dataset.get_item_list)
                for element_id, element in enumerate(dataloader.dataset.get_item_list):
                    batch_dict[element] = batch[element_id]
                batch = edict(batch_dict)

            # batch = batch.to(args.device)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device)
            
            # Forward pass
            recon_data, loss_commit, indices, gt_data = vqvae(batch)

            """
            viz_skel_seq_anim({'recon':recon_data[0], 'gt':batch[0]},fs=0.5,subplot_layout=(1,2),if_print=1,file_folder='.',lim3d=0.5)
            """

            # Update codebook usage
            if indices is not None:
                unique_indices, counts = torch.unique(indices, return_counts=True)
                codebook_usage.scatter_add_(0, unique_indices, counts)
            
            # Ensure batch and recon_data have same length for loss calculation
            min_len = min(gt_data.shape[1], recon_data.shape[1])



            if args.loss_type == 'mpjpe_millimeter':

                if args.joint_data_type == 'joint3d_cam_rootrel_meter':
                    recon_data = (recon_data - recon_data[..., 0:1, :]) * 1000.0  # to millimeter
                    gt_data = (gt_data - gt_data[..., 0:1, :]) * 1000.0                    

                else:
                    if args.joint_data_type == 'joint3d_image_affined_normed':
                        recon_data_affined = (recon_data + batch.joint3d_image_affined_transl[..., None, :]) * batch.joint3d_image_affined_scale[..., None, :]
                        recon_data_affined_xy = recon_data_affined[..., :2].clone()   # [B,T,17,2]
                        trans_inv = batch.affine_trans_inv   # [B,T,2,3]
                        recon_data_affined_xy1 = torch.cat([recon_data_affined_xy, torch.ones_like(recon_data_affined_xy[..., :1])], dim=-1)# [B,T,17,3]
                        recon_data_3dimage_xy = torch.einsum('btij,btkj->btki', trans_inv, recon_data_affined_xy1)# [B,T,17,2]
                        recon_data_3dimage = torch.cat([recon_data_3dimage_xy, recon_data_affined[..., 2:]], dim=-1)# [B,T,17,3]

                    elif args.joint_data_type == 'joint3d_image_normed':
                        recon_data_3dimage = (recon_data + batch.joint3d_image_transl[..., None, :]) * batch.joint3d_image_scale[..., None, :]

                    factor_2_5d = batch.factor_2_5d[..., None, None]  # [B,1,1]

                    recon_data_2_5dimage = recon_data_3dimage * factor_2_5d
                    gt_data_2_5dimage = batch.joint_2_5d_image  # [B,T,17,3]

                    recon_data_2_5dimage_rootrel = recon_data_2_5dimage - recon_data_2_5dimage[..., 0:1, :]
                    gt_data_2_5dimage_rootrel = gt_data_2_5dimage - gt_data_2_5dimage[..., 0:1, :]

                    mpjpe_millimeter = torch.norm(recon_data_2_5dimage_rootrel - gt_data_2_5dimage_rootrel, dim=-1).mean((-2, -1))


                    recon_data = recon_data_2_5dimage_rootrel
                    gt_data = gt_data_2_5dimage_rootrel


            # Calculate loss
            reconstruction_loss = loss_fn(recon_data[:, :min_len], gt_data[:, :min_len])
            total_l1_loss += reconstruction_loss.item()



    # --- 全局同步 codebook_usage ---
    # 所有进程的 codebook_usage 累加到主进程
    accelerator.wait_for_everyone()
    global_codebook_usage = codebook_usage.clone()
    accelerator.reduce(global_codebook_usage, reduction="sum")

    if accelerator.is_main_process:
        # --- 5. Analyze Codebook Usage ---
        print("\n--- Codebook Usage Statistics ---")
        # total_codes_used = codebook_usage.cpu().sum().item()
        # num_codes_activated = (codebook_usage > 0).sum().item()
        global_codebook_usage = global_codebook_usage.cpu()
        total_codes_used = global_codebook_usage.sum().item()
        num_codes_activated = (global_codebook_usage > 0).sum().item()
        num_codes_total = args.nb_code
        usage_percentage = (num_codes_activated / num_codes_total) * 100
        
        print(f"Total codes selected: {total_codes_used}")
        print(f"Codebook size: {num_codes_total}")
        print(f"Number of activated codes: {num_codes_activated} ({usage_percentage:.2f}%)")

        # Top 5 most used codes
        top_k = min(5, num_codes_activated)
        if top_k > 0:
            # top_counts, top_indices = torch.topk(codebook_usage, k=top_k)
            top_counts, top_indices = torch.topk(global_codebook_usage, k=top_k)
            print(f"\nTop {top_k} most used codes:")
            for i in range(top_k):
                print(f"  - Code {top_indices[i].item()}: used {top_counts[i].item()} times")

        # Top 5 least used (but still used > 0)
        # used_mask = codebook_usage > 0
        used_mask = global_codebook_usage > 0
        if used_mask.any():
            # used_codes_counts = codebook_usage[used_mask]
            used_codes_counts = global_codebook_usage[used_mask]
            used_codes_indices = torch.arange(num_codes_total, device='cpu')[used_mask]
            
            bottom_k = min(5, len(used_codes_counts))
            if bottom_k > 0:
                bottom_counts, bottom_indices_of_used = torch.topk(used_codes_counts, k=bottom_k, largest=False)
                original_indices = used_codes_indices[bottom_indices_of_used]
                
                print(f"\nTop {bottom_k} least used codes (among activated):")
                for i in range(bottom_k):
                    print(f"  - Code {original_indices[i].item()}: used {bottom_counts[i].item()} times")




    avg_l1_loss = total_l1_loss / len(dataloader)
    print(f"\n--- Test Results ---")
    print(f"Average {args.loss_type} Reconstruction Loss: {avg_l1_loss:.6f}")




    # if args.num_vis_samples > 0:
    #     print(f"\nVisualizations saved in: {args.output_dir}")




if __name__ == '__main__':
    args = get_args()
    test_vqvae(args)
