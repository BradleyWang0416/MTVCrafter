import os
import gc
import copy
import torch
# import decord
import pickle
import random
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from concurrent.futures import ThreadPoolExecutor



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


class SMPLDataset(torch.utils.data.Dataset):
    def __init__(self, data_root=None, num_frames=25, dst_size=(512, 512), load_data_file=""):
        self.data_root = data_root
        self.save_data_root = './data'
        self.num_frames = num_frames
        self.dst_size = dst_size
        self.data_list = []
        if load_data_file == "":
            self.load_data()
        else:
            with open(load_data_file, 'rb') as f:
                self.data_list = pickle.load(f)
        self.global_mean, self.global_std = self.calculate_global_mean_std()
        self.global_min, self.global_max = self.calculate_global_min_max()
        self.total_frames = sum([data['video_length'] for data in self.data_list])
        self.normalize = transforms.Normalize([0.5], [0.5])

    def load_data(self):
        batch_size = 100
        temp_data_list = []
        flag = False
        for root, dirs, files in os.walk(self.data_root):
            for dir_name in dirs:
                if 'smpl' in dir_name:
                    part = 0
                    folder_path = os.path.join(root, dir_name)
                    print(f"loading {folder_path}")
                    data_list = os.listdir(folder_path)
                    if data_list:
                        self.process_data_list = data_list
                        self.temp_folder_path = folder_path
                        num = min(len(data_list), 1e9)
                        for start_idx in range(part*batch_size, num, batch_size):
                            end_idx = min(start_idx + batch_size, num)
                            print(f"Processing data {start_idx} to {end_idx}")
                            with ThreadPoolExecutor() as executor:
                                results = list(tqdm(
                                    executor.map(self._process_video, range(start_idx, end_idx)),
                                    total=end_idx - start_idx,
                                    desc="Processing data",
                                    unit="video"
                                ))
                            temp_data_list.extend([result for result in results if result is not None])
                            self.save_temp_data(temp_data_list, folder_path, part)
                            part += 1
                            temp_data_list = []
                            del results 
                            gc.collect()

                    self.data_list = self.merge_temp_data(folder_path)   
                    gc.collect()
    
    def save_temp_data(self, data, folder_path, part):
        spe = folder_path.split('/')[-1]
        os.makedirs(f'{self.save_data_root}/{spe}', exist_ok=True)
        output_file = f'{self.save_data_root}/{spe}/temp_data_part_{part}.pkl'
        with open(output_file, 'wb') as file:
            pickle.dump(data, file)
        print(f"save {spe} temp data to {output_file}")

    def merge_temp_data(self, folder_path):
        merged_data = []
        part = 0
        spe = folder_path.split('/')[-1]
        while True:
            temp_file = f'{self.save_data_root}/{spe}/temp_data_part_{part}.pkl'
            if not os.path.exists(temp_file):
                break
            with open(temp_file, 'rb') as file:
                temp_data = pickle.load(file)
                merged_data.extend(temp_data)
            part += 1
        # return merged_data
        merge_file = f'{self.save_data_root}/{spe}/{spe}.pkl'
        with open(merge_file, 'wb') as file:
            pickle.dump(merged_data, file)
            print(f"merge {spe} temp data to {merge_file}")
        del merged_data
        

    def _process_video(self, idx):
        data_file = self.process_data_list[idx]
        file_path = os.path.join(self.temp_folder_path, data_file)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            try:
                smpl_poses = data['poses']['joints3d_nonparam']
            except Exception as e:
                print(f"Error loading pose data {file_path}: {e}")
                return None
            frame_count = data['frame_count']

            if frame_count >= self.num_frames and sum(len(poses) for poses in smpl_poses) == frame_count and all(len(poses[0]) > 0 for poses in smpl_poses): 
                # print(idx, file_path)
                result = {
                    'file_path': file_path,
                    'video_path': data['video_path'],
                    # 'video': video,
                    'pose': np.array([poses[0][0].cpu().numpy() for poses in smpl_poses]),
                    'video_length': frame_count,
                    'video_height': data['video_height'],
                    'video_width': data['video_width']
                }
                return result
            else:
                print(idx, video_path, "skipped")
                return None


    def calculate_global_mean_std(self):
        pose_list = [data['pose'] if 'pose' in data else data['joint'] for data in self.data_list]
        all_frames = np.concatenate(pose_list, axis=0).astype(np.float32)  # shape: [total_frames, 24, 3]
        mean = np.mean(all_frames, axis=0)  # shape: [24, 3]
        std = np.std(all_frames, axis=0)  # shape: [24, 3]
        return mean, std
    
    def calculate_global_min_max(self):
        pose_list = [data['pose'] if 'pose' in data else data['joint'] for data in self.data_list]
        all_frames = np.concatenate(pose_list, axis=0).astype(np.float32)  # shape: [total_frames, 24, 3]
        min = np.min(all_frames, axis=(0, 1))  # shape: [3]
        max = np.max(all_frames, axis=(0, 1))  # shape: [3]
        # min = np.min(all_frames, axis=0)     # shape: [24, 3]
        # max = np.max(all_frames, axis=0)     # shape: [24, 3]
        return min, max


    def get_sample_indexes(self, data_dict, num_frames):
        video_length = data_dict['video_length']
        sample_length = min(video_length, (num_frames - 1) + 1)
        start_idx = random.randint(0, video_length - sample_length)
        sample_indexes = np.linspace(start_idx, start_idx + sample_length - 1, num_frames, dtype=int)
        return sample_indexes


    def get_sample_indexes_from_start(self, data_dict, num_frames):
        video_length = data_dict['video_length']
        sample_length = min(video_length, (num_frames - 1) + 1)
        start_idx = data_dict['video_start_index'] + random.randint(0, video_length - sample_length)
        sample_indexes = np.linspace(start_idx, start_idx + sample_length - 1, num_frames, dtype=int)
        return sample_indexes
    
    def get_new_height_width(self, data_dict):
        if 'video_height' in data_dict:
            height = data_dict['video_height']
            width = data_dict['video_width']
        dst_width, dst_height = self.dst_size
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        assert dst_width <= new_width and dst_height <= new_height
        return new_height, new_width, dst_height, dst_width

    def __len__(self):
        return len(self.data_list)

    # use for MV-DiT training
    # def __getitem__(self, idx):
    #     video_path = self.data_list[idx]['video_path'].replace('/nvfile-heatstorage', '/gemini-1/space')
    #     while not os.path.exists(video_path):
    #         idx = random.randint(0, len(self.data_list) - 1)
    #         video_path = self.data_list[idx]['video_path'].replace('/nvfile-heatstorage', '/gemini-1/space')

    #     data_dict = self.data_list[idx]
    #     new_height, new_width, dst_height, dst_width = self.get_new_height_width(data_dict)
    #     x1 = random.randint(0, new_width - dst_width)
    #     y1 = random.randint(0, new_height - dst_height)

    #     video_path = data_dict['video_path'].replace('/nvfile-heatstorage', '/gemini-1/space')
    #     sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
    #     input_images = sample_video(decord.VideoReader(video_path), sample_indexes, method=2)
    #     input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
    #     input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
    #     input_images = F.crop(input_images, y1, x1, dst_height, dst_width)

    #     ref_images = copy.deepcopy(input_images)
    #     frame0 = input_images[0]
    #     ref_images[:, :, :, :] = frame0
    #     input_images = input_images / 255.0
    #     ref_images = ref_images / 255.0

    #     input_images = self.normalize(input_images)
    #     ref_images = self.normalize(ref_images)
    #     norm_poses = torch.tensor((data_dict['pose'][sample_indexes] - self.global_mean) / self.global_std)
    #     result = {
    #         'images': input_images,
    #         'ref_images': ref_images,
    #         'poses': norm_poses,
    #     }

    #     return result

    # use for 4DMoT training
    def __getitem__(self, idx):
        video_path = self.data_list[idx]['video_path']
        while not os.path.exists(video_path):
            idx = random.randint(0, len(self.data_list) - 1)
            video_path = self.data_list[idx]['video_path']

        data_dict = self.data_list[idx]
        if 'video_start_index' in data_dict:
            sample_indexes = self.get_sample_indexes_from_start(data_dict, self.num_frames)
            poses = data_dict['joint'][sample_indexes-data_dict['video_start_index']].astype(np.float32)
            # norm_poses = torch.tensor((poses - self.global_mean) / self.global_std)
        else:
            sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
            poses = data_dict['pose'][sample_indexes].astype(np.float32)
            # norm_poses = torch.tensor((poses - self.global_mean) / self.global_std)
        
        # mean = np.mean(poses, axis=(0, 1))  # (3,)
        # std = np.std(poses, axis=(0, 1))    # (3,)
        # min = np.min(poses, axis=(0, 1))    # (3,)
        # max = np.max(poses, axis=(0, 1))    # (3,)
        # norm_poses_1 = torch.tensor((poses - min) / (max - min))
        # norm_poses_2 = torch.tensor((poses - mean) / std)
        # norm_poses_3 = torch.tensor((poses - self.global_mean) / (self.global_max - self.global_min))
        norm_poses_4 = torch.tensor((poses - self.global_mean) / self.global_std)

        return norm_poses_4
