import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
import yaml
from easydict import EasyDict as edict
import ast
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, release_memory, ProjectConfiguration, DistributedDataParallelKwargs
from safetensors.torch import load_file as load_safetensors

from config.vision_backbone import config as vision_config
from config.vqvae import vqvae_config
# from models import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder
from models import HYBRID_VQVAE
# from dataset_byBradley import SkeletonDataset

import sys
sys.path.append("../Skeleton-in-Context-tpami/")
from lib.utils.viz_skel_seq import viz_skel_seq_anim
sys.path.remove("../Skeleton-in-Context-tpami/")
sys.path.append("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset
sys.path.remove("../ContextAwarePoseFormer_Private/H36M-Toolbox/")


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
    
    parser.add_argument('--resume_pth', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--commit_ratio', type=float, default=0.5)
    parser.add_argument('--nb_code', type=int, default=8192)
    parser.add_argument('--codebook_dim', type=int, default=3072)

    parser.add_argument('--max_epoch', type=int, default=1e9)
    parser.add_argument('--total_iter', type=int, default=500000)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=20000)
    parser.add_argument('--warm_up_iter', type=int, default=5000)
    parser.add_argument('--print_iter', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--lr_schedule', type=list, default=[300000])
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--project_config', type=str, default='')
    parser.add_argument('--allow_tf32', action='store_true')
    parser.add_argument('--project_dir', type=str, default='./vqvae_experiment')
    parser.add_argument('--seed', type=int, default=6666)

    parser.add_argument('--not_find_unused_parameters', action='store_true')

    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mpjpe'])

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
    parser.add_argument('--fix_weights', action='store_true')
    parser.add_argument('--fix_weights_except', type=str, default='PLACEHOLDERPLACEHOLDERPLACEHOLDER')
    parser.add_argument('--vision_guidance_ratio', type=float, default=None)

    parser.add_argument('--downsample_time', type=str, default=None)
    parser.add_argument('--frame_upsample_rate', type=str, default=None)

    parser.add_argument('--vision_guidance_where', type=str, default=None)
    parser.add_argument('--vision_guidance_fuse', type=str, default=None)
    parser.add_argument('--vision_guidance_extraLoss', type=str, default=None)

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


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def train_vqvae(
    args,
    accelerator, 
    logger, 
    vqvae, 
    optimizer, 
    dataloader, 
    scheduler, 
    save_interval=10000,
    warm_up_iter=2000,
    learning_rate=2e-4,
    print_iter=200,
    total_iter=5e5,
    commit_ratio=0.25,
    max_epoch=1e9, 
    save_dir='',
    resume_iter=0,
    resume_epoch=0,
    train_sampler=None, 
    ):
    
    if accelerator.is_main_process:
        logger.info('Args: {}'.format(args))

    recon_loss = 0
    commit_loss = 0
    total_loss = 0
    if args.get('vision_guidance_extraLoss', None) is not None:
        EXTRA_LOSS = 0
    total_preplexity = 0
    nb_iter = resume_iter
    epoch_start = resume_epoch

    for epoch in tqdm(range(max_epoch), desc='Trainning Epoch', initial=epoch_start, position=0):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            
            if isinstance(batch, tuple):
                batch_dict = {}
                assert len(batch) == len(dataloader.dataset.get_item_list)
                for element_id, element in enumerate(dataloader.dataset.get_item_list):
                    batch_dict[element] = batch[element_id]
                batch = edict(batch_dict)

            nb_iter += 1
            if nb_iter <= warm_up_iter:
                stage = 'Warm Up'
            else:
                stage = f'Train {commit_ratio}'

            if stage == 'Warm Up':
                optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, warm_up_iter, learning_rate)
                if nb_iter % print_iter == 0 and accelerator.is_main_process:
                    logger.info(f'current_lr {current_lr:.6f} at iteration {nb_iter}')


            tuple_return  = vqvae(batch)
            # [B,T,17,3]
            if args.get('vision_guidance_extraLoss', None) is not None:
                recon_data, loss_commit, perplexity, gt_data, extra_loss = tuple_return
            else:
                recon_data, loss_commit, perplexity, gt_data = tuple_return



            if args.loss_type == 'l1':
                loss_fn = nn.L1Loss()
            elif args.loss_type == 'mpjpe':
                def mpjpe_loss(pred, target):
                    return torch.mean(torch.norm(pred - target, dim=-1))
                loss_fn = mpjpe_loss


            reconstruction_loss = loss_fn(recon_data, gt_data)
            loss = reconstruction_loss + commit_ratio * loss_commit
            if args.get('vision_guidance_extraLoss', None) is not None:
                loss = loss + extra_loss    # loss weight applied in forward
                EXTRA_LOSS += extra_loss.item()
            recon_loss += reconstruction_loss.item()
            commit_loss += loss_commit.item()
            total_loss += loss.item()
            total_preplexity += perplexity.item()

            # backward and optimize
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if stage == 'Train':
                scheduler.step()
            torch.cuda.synchronize()

            if nb_iter % print_iter == 0 and accelerator.is_main_process:
                if args.get('vision_guidance_extraLoss', None) is not None:
                    logger.info(f"Stage: {stage} | Epoch: {epoch} | Iter: {nb_iter} | Total Loss: {(total_loss / print_iter):.6f} | Recon Loss: {(recon_loss / print_iter):.6f} | Commit Loss: {(commit_loss / print_iter):.6f} | Perplexity: {(total_preplexity / print_iter):.6f} | {args.get('vision_guidance_extraLoss', None)}: {(EXTRA_LOSS / print_iter):.6f}")
                    total_loss, recon_loss, commit_loss, total_preplexity = 0, 0, 0, 0
                    EXTRA_LOSS = 0
                else:
                    logger.info(f'Stage: {stage} | Epoch: {epoch} | Iter: {nb_iter} | Total Loss: {(total_loss / print_iter):.6f} | Recon Loss: {(recon_loss / print_iter):.6f} | Commit Loss: {(commit_loss / print_iter):.6f} | Perplexity: {(total_preplexity / print_iter):.6f}')
                    total_loss, recon_loss, commit_loss, total_preplexity = 0, 0, 0, 0
            if nb_iter % save_interval == 0:
                if accelerator.is_main_process:
                    logger.info('Saving model at iteration {}'.format(nb_iter))
                output_name = 'checkpoint_epoch_{}_step_{}'.format(epoch+1, nb_iter)
                output_dir = os.path.join(save_dir, output_name)
                if os.path.exists(output_dir) and accelerator.is_main_process:
                    import shutil
                    shutil.rmtree(output_dir)
                release_memory()
                accelerator.wait_for_everyone()
                accelerator.save_state(output_dir)

            if nb_iter >= total_iter:
                break

        if nb_iter >= total_iter:
            break


def create_logger(log_path=None, log_format=None):
    import logging

    if log_format is None:
        log_format = '%(asctime)-15s %(message)s'
    if log_path is not None:
        if os.path.exists(log_path):
            os.remove(log_path)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler_s = logging.StreamHandler()
        handler_s.setFormatter(formatter)
        logger.addHandler(handler_s)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)

    return logger


def init_env(args):
    assert args.seed > 0
    set_seed(args.seed)

    project_config = ProjectConfiguration(
        project_dir=args.project_dir,
        logging_dir=os.path.join(args.project_dir, 'logs'),
    )

    accelerator = Accelerator(
        log_with='tensorboard',
        project_config=project_config,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters= False if args.not_find_unused_parameters else True,
                broadcast_buffers=True,
            ),
        ],
    )

    if args.project_dir.endswith('/'):
        args.project_dir = args.project_dir[:-1]
    else:
        args.project_dir = args.project_dir
    project_name = os.path.basename(args.project_dir)

    accelerator.init_trackers(project_name)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return accelerator


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()

    # initialize environment and logger
    accelerator = init_env(args)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.project_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(args.project_dir, 'models'), exist_ok=True)
        log_name = 'train_{}.log'.format(time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())))
        logger = create_logger(os.path.join(os.path.join(args.project_dir, 'logs'), log_name))
    else:
        logger = None

    try:
        logger.info('\npython ' + ' '.join(sys.argv))
        logger.info('\nPID: '+str(os.getpid()))
    except:
        pass

    # prepare data
    # dataset = SkeletonDataset(num_frames=args.num_frames, sample_stride=args.sample_stride, load_data_file=args.load_data_file, data_mode=args.data_mode)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=0,
    #     shuffle=True,
    #     drop_last=False
    #     )
    dataset = Multimodal_Mocap_Dataset( num_frames=args.num_frames, sample_stride=args.sample_stride, data_stride=args.data_stride,
                                        data_mode=args.data_mode,
                                        designated_split='train',
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
                                        max_samples=512 if 'debugpy' in sys.modules else None,  # for debug
                                        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    if accelerator.is_main_process:
        logger.info('Data loaded with {} samples'.format(len(dataset)))

    # prepare model
    vqvae_config.encoder.out_channels = args.codebook_dim
    vqvae_config.vq.nb_code = args.nb_code
    vqvae_config.vq.code_dim = args.codebook_dim
    vqvae_config.decoder.in_channels = args.codebook_dim

    if args.get('downsample_time', None) is not None:
        vqvae_config.encoder.downsample_time = args.downsample_time
    if args.get('frame_upsample_rate', None) is not None:
        vqvae_config.decoder.frame_upsample_rate = args.frame_upsample_rate

    # if args.vqvae_type == 'base':
    #     encoder = Encoder(**vqvae_config.encoder)
    #     vq = VectorQuantizer(**vqvae_config.vq)
    #     decoder = Decoder(**vqvae_config.decoder)
    #     vqvae = SMPL_VQVAE(encoder, decoder, vq).train()
    # elif args.vqvae_type == 'hybrid':
    if args.get('hrnet_output_level', None) is not None:
        vision_config.model.hybrid.hrnet_output_level = args.hrnet_output_level
    if args.get('vision_guidance_ratio', None) is not None:
        vision_config.model.hybrid.vision_guidance_ratio = args.vision_guidance_ratio
    if args.get('vision_guidance_where', None) is not None:
        vision_config.model.hybrid.vision_guidance_where = args.vision_guidance_where
    if args.get('vision_guidance_fuse', None) is not None:
        vision_config.model.hybrid.vision_guidance_fuse = args.vision_guidance_fuse
    if args.get('vision_guidance_extraLoss', None) is not None:
        vision_config.model.hybrid.vision_guidance_extraLoss = args.get('vision_guidance_extraLoss', None)
        


    vqvae = HYBRID_VQVAE(vqvae_config.encoder, vqvae_config.decoder, vqvae_config.vq, vision_config=vision_config, joint_data_type=args.joint_data_type).train()
    
    if vision_config.model.hybrid.vision_guidance_ratio > 0 or vision_config.model.hybrid.vision_guidance_ratio == -1:
        if args.get('fix_weights', None) is not None:
            vision_config.model.backbone.fix_weights = args.fix_weights
        if vision_config.model.backbone.fix_weights:
            print("vision backbone weights are fixed")


            
            fix_weights_except = args.fix_weights_except.split(',')


            for name, p in vqvae.vision_backbone.named_parameters():
                if any(except_str in name for except_str in fix_weights_except):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            
            if all(not p.requires_grad for p in vqvae.vision_backbone.parameters()):
                vqvae.vision_backbone.eval()

    # 统计可学习和不可学习参数量
    def count_parameters(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable, non_trainable
    if accelerator.is_main_process:
        trainable_params, non_trainable_params = count_parameters(vqvae)
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {non_trainable_params:,}")

    # prepare optimizer and scheduler
    optimizer = optim.AdamW(vqvae.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=args.gamma)

    # move all to accelerator
    vqvae, optimizer, dataloader, scheduler = accelerator.prepare(vqvae, optimizer, dataloader, scheduler)

    # resume checkpoint
    if args.resume_pth != '':
        resume_iter = int(args.resume_pth.split('_')[-1])
        resume_epoch = int(args.resume_pth.split('_')[-3])
        if accelerator.is_main_process:
            logger.info('Loading checkpoint from {}'.format(args.resume_pth))
            logger.info('Resuming from epoch {} and iteration {}'.format(resume_epoch, resume_iter))
        # try:
        state_dict = load_safetensors(os.path.join(args.resume_pth, "model.safetensors"), device="cpu")
        try: 
            missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=True)
        except Exception as e:
            state_dict = {'module.'+k: v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=True)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        # except:
        #     accelerator.load_state(args.resume_pth)
        # else:
        #     state_dict = torch.load(args.resume_pth, map_location="cpu")
        #     missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=False)
        #     print(f"Missing keys: {missing_keys}")
        #     print(f"Unexpected keys: {unexpected_keys}")
    else:
        resume_iter = 0
        resume_epoch = 0

    # training
    if accelerator.is_main_process:
        n = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {n/1e6:.6f} M')

    train_vqvae(
        args,
        accelerator, 
        logger, 
        vqvae, 
        optimizer, 
        dataloader, 
        scheduler, 
        args.save_interval,
        args.warm_up_iter,
        args.learning_rate,
        args.print_iter,
        args.total_iter,
        args.commit_ratio, 
        max_epoch=math.ceil(args.total_iter / math.ceil(len(dataset) / accelerator.num_processes / args.batch_size)), 
        save_dir=os.path.join(args.project_dir, 'models'),
        resume_iter=resume_iter,
        resume_epoch=resume_epoch,
        # train_sampler=train_sampler, 
    )
    if accelerator.is_main_process:
        logger.info('Training finished')


    
