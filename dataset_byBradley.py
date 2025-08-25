import torch
import numpy as np
import joblib

import sys
sys.path.append("/home/wxs/Skeleton-in-Context-tpami/")
from funcs_and_classes.Non_AR.dataset.ver13_ICL import DataReaderMesh

def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        frames = frames[indexes]
    else:
        assert False
    return frames


class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=25, load_data_file="", data_mode="joint2d", designated_split='train'):
        self.num_frames = num_frames

        assert load_data_file != ""
        datareader_config_unsplit = {
            'dt_file': load_data_file,
        }
        datareader_config_split = {
            'chunk_len': num_frames,
            'sample_stride': 1, 
            'data_stride': num_frames,
            'read_confidence': False,
        }
        datareader_config = {}
        datareader_config.update({'read_modality': ['joint2d', 'joint3d']})
        datareader_config.update(**datareader_config_unsplit, **datareader_config_split)
        datareader = DataReaderMesh(**datareader_config)        
        unsplit_data = DataReaderMesh.load_dataset_static(**datareader_config_unsplit)   # '/data2/wxs/DATASETS/AMASS_ByBradley'
        datareader.dt_dataset = unsplit_data

        read_func = datareader.read_2d if data_mode == "joint2d" else datareader.read_3d_image
        self.all_data = read_func(designated_split=designated_split)
        self.total_frames = self.all_data.shape[0]
        self.split_id = datareader.get_split_id(designated_split=designated_split)

        self.global_mean, self.global_std = self.calculate_global_mean_std()
        self.global_min, self.global_max = self.calculate_global_min_max()

    def calculate_global_mean_std(self):
        mean = np.mean(self.all_data, axis=0)  # shape: [24, 3]
        std = np.std(self.all_data, axis=0)  # shape: [24, 3]
        return mean, std
    
    def calculate_global_min_max(self):
        min = np.min(self.all_data, axis=(0, 1))  # shape: [3,]
        max = np.max(self.all_data, axis=(0, 1))  # shape: [3,]
        return min, max

    def __len__(self):
        return len(self.split_id)

    # use for 4DMoT training
    def __getitem__(self, idx):
        slice_id = self.split_id[idx]
        poses = self.all_data[slice_id]

        # norm_poses_4 = torch.tensor((poses - self.global_mean) / self.global_std)
        poses = torch.from_numpy(poses).float()
        # 已经经过screen coord normalize了

        return poses
