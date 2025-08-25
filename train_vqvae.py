import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, release_memory, ProjectConfiguration, DistributedDataParallelKwargs

from dataset import SMPLDataset
# BY BRADLEY
from dataset_byBradley import SkeletonDataset
from models import SMPL_VQVAE, VectorQuantizer, Encoder, Decoder


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--num_frames', type=int, default=49)
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--resume_pth', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--project_config', type=str, default='')
    parser.add_argument('--allow_tf32', action='store_true')
    parser.add_argument('--project_dir', type=str, default='./vqvae_experiment')
    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--commit_ratio', type=float, default=0.5)
    parser.add_argument('--nb_code', type=int, default=8192)
    parser.add_argument('--codebook_dim', type=int, default=3072)

    parser.add_argument('--load_data_file', type=str, default="./data/train_data.pkl")
    # BY BRADLEY
    parser.add_argument('--data_mode', type=str, default="")
    parser.add_argument('--not_find_unused_parameters', action='store_true')

    return parser.parse_args()


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def train_vqvae(
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
    total_preplexity = 0
    nb_iter = resume_iter
    epoch_start = resume_epoch

    for epoch in tqdm(range(max_epoch), desc='Trainning Epoch', initial=epoch_start, position=0):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            nb_iter += 1
            if nb_iter <= warm_up_iter:
                stage = 'Warm Up'
            else:
                stage = f'Train {commit_ratio}'

            if stage == 'Warm Up':
                optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, warm_up_iter, learning_rate)
                if nb_iter % print_iter == 0 and accelerator.is_main_process:
                    logger.info(f'current_lr {current_lr:.6f} at iteration {nb_iter}')


            recon_data, loss_commit, perplexity = vqvae(batch)
            reconstruction_loss = nn.L1Loss()(recon_data, batch)
            loss = reconstruction_loss + commit_ratio * loss_commit
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

    logger.info('\npython ' + ' '.join(sys.argv))

    # prepare data
    # BY BRADLEY ST ####################################
    if args.data_mode:
        assert args.load_data_file != ""
        dataset = SkeletonDataset(num_frames=args.num_frames, load_data_file=args.load_data_file, data_mode=args.data_mode)
    else:
    # BY BRADLEY ED ####################################
        if args.load_data_file != "":
            dataset = SMPLDataset(load_data_file=args.load_data_file, num_frames=args.num_frames)
            if accelerator.is_main_process:
                logger.info('Global mean:\n {}'.format(dataset.global_mean))
                logger.info('Global std:\n {}'.format(dataset.global_std))
                param = {
                    "mean": dataset.global_mean,
                    "std": dataset.global_std,
                }
        else:
            dataset = SMPLDataset(data_root=args.data_root, num_frames=args.num_frames)
            if accelerator.is_main_process:
                logger.info('Global mean:\n {}'.format(dataset.global_mean))
                logger.info('Global std:\n {}'.format(dataset.global_std))
                

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=False
        )
    if accelerator.is_main_process:
        logger.info('Data loaded with {} samples and {} frames'.format(len(dataset), dataset.total_frames))

    # prepare model
    encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=args.codebook_dim, downsample_time=[2, 2], downsample_joint=[1, 1])
    vq = VectorQuantizer(nb_code=args.nb_code, code_dim=args.codebook_dim)
    decoder = Decoder(in_channels=args.codebook_dim, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    vqvae = SMPL_VQVAE(encoder, decoder, vq).train()

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
        try:
            accelerator.load_state(args.resume_pth)
        except:
            state_dict = torch.load(args.resume_pth, map_location="cpu")
            missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
    else:
        resume_iter = 0
        resume_epoch = 0

    # training
    if accelerator.is_main_process:
        n = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {n/1e6:.6f} M')

    train_vqvae(
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


    
