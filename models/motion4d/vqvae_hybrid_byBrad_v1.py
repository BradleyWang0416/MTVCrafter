import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file as load_safetensors
import math
from collections import defaultdict
from PIL import Image

from .vqvae import SMPL_VQVAE, ResBlock, Downsample, Encoder, Decoder, VectorQuantizer
from .conditional_decoder_byBrad import ConditionalDecoder

import sys
sys.path.append('../ContextAware-PoseFormer/ContextPose/mvn/models/')
import pose_hrnet
sys.path.remove('../ContextAware-PoseFormer/ContextPose/mvn/models/')

class VisionEncoder(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        mid_channels=[128, 512], 
        out_channels=3072,
        downsample_time=[1, 1],
        downsample_joint=[1, 1],
        num_attention_heads=8,
        attention_head_dim=64,
        dim=3072,
        ):
        super(VisionEncoder, self).__init__()

        # self.conv_in = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.downsample1 = Downsample(mid_channels[0], mid_channels[0], downsample_time[0], downsample_joint[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.downsample2 = Downsample(mid_channels[1], mid_channels[1], downsample_time[1], downsample_joint[1])
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # x = self.conv_in(x) # [B,3,8,17] -> [B,128,8,17]
        for resnet in self.resnet1:
            x = resnet(x)   # 不改变形状
        x = self.downsample1(x) # [B,128,8,17] -> [B,128,4,17]
        
        x = self.resnet2(x) # [B,128,4,17] -> [B,512,4,17]
        for resnet in self.resnet3:
            x = resnet(x)   # 不改变形状
        x = self.downsample2(x) # [B,512,4,17] -> [B,512,2,17]

        x = self.conv_out(x)    # [B,512,2,17] -> [B,out_channels,2,17]

        return x


class AdaptiveSampler(nn.Module):
    def __init__(self, feature_channels, num_sampling_points=4, merge_sampled_points_w_attn=False):
        """
        Args:
            feature_channels (int): 输入特征图的通道数。
            num_sampling_points (int): 在每个关键点周围采样的点的数量。这直接解决了你的 TODO 问题。
        """
        super().__init__()
        self.num_sampling_points = num_sampling_points
        
        # 一个小型网络，用于预测偏移量。
        # 它以原始关键点位置的特征作为输入，
        # 输出每个采样点的 2D 偏移量。
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(feature_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            # 输出 N_points * 2 个值，分别代表 N 个点的 (dx, dy)
            nn.Conv2d(128, num_sampling_points * 2, kernel_size=1)
        )

        self.merge_sampled_points_w_attn = merge_sampled_points_w_attn
        if self.merge_sampled_points_w_attn:
            self.attention_predictor = nn.Sequential(
                nn.Conv2d(feature_channels, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_sampling_points, kernel_size=1)
            )
        
        # 将最后一层的权重和偏置初始化为 0。
        # 这确保在训练开始时，预测的偏移量为 0，
        # 使得模型的初始行为与标准的 grid_sample 相同，有助于稳定训练。
        nn.init.constant_(self.offset_predictor[-1].weight, 0.)
        nn.init.constant_(self.offset_predictor[-1].bias, 0.)
        if self.merge_sampled_points_w_attn:
            nn.init.constant_(self.attention_predictor[-1].weight, 0.) # <--- 新增
            nn.init.constant_(self.attention_predictor[-1].bias, 0.)   # <--- 新增

    def forward(self, features, keypoint_coords):
        """
        Args:
            features (torch.Tensor): 图像特征图，形状为 [BT, C, H, W]。
            keypoint_coords (torch.Tensor): 原始关键点坐标，已归一化到 [-1, 1] 范围，
                                           形状为 [BT, J, 2]。
        
        Returns:
            torch.Tensor: 采样后的特征，形状为 [BT, J, C * N_points]。
        """
        BT, C, H, W = features.shape
        J = keypoint_coords.shape[1]
        
        # 1. 采样初始特征以预测偏移量
        #    keypoint_coords: [BT, J, 2] -> grid: [BT, J, 1, 2]
        initial_grid = keypoint_coords.unsqueeze(-2)
        # seed_features: [BT, C, J, 1]
        seed_features = F.grid_sample(features, initial_grid, align_corners=True)
        
        # 2. 预测偏移量
        #    seed_features: [BT, C, J, 1] -> [BT, J, C] -> [BT, C, J, 1] for Conv2d
        #    offsets: [BT, N_points*2, J, 1]
        offsets = self.offset_predictor(seed_features)


        if self.merge_sampled_points_w_attn:
            attn_logits = self.attention_predictor(seed_features)



        
        # 3. 整理偏移量并创建新的采样格点
        #    offsets: [BT, N_points*2, J, 1] -> [BT, J, N_points*2] -> [BT, J, N_points, 2]
        offsets = offsets.squeeze(-1).permute(0, 2, 1).reshape(BT, J, self.num_sampling_points, 2)
        
        # 注意: 预测的偏移量是在特征图尺度上的，需要归一化到 [-1, 1] 范围
        # 一个单位的偏移量对应于特征图上的一个像素
        # 归一化公式: offset_norm = offset_pixel * (2.0 / (grid_size - 1))
        # 这里我们简化处理，乘以 2 / (H or W)，对于非方形特征图更严谨的做法是分开处理 x 和 y
        norm_offsets = offsets.clone()
        norm_offsets[..., 0] = norm_offsets[..., 0] / (W - 1) * 2
        norm_offsets[..., 1] = norm_offsets[..., 1] / (H - 1) * 2
        
        # 新的采样格点 = 原始格点 + 预测的归一化偏移量
        # 原始格点 [BT, J, 1, 2] 需要广播以匹配偏移量 [BT, J, N_points, 2]
        new_grid = keypoint_coords.unsqueeze(-2) + norm_offsets # [BT, J, N_points, 2]
        
        # 4. 使用新的自适应格点进行最终采样
        #    sampled_features: [BT, C, J, N_points]
        sampled_features = F.grid_sample(features, new_grid, align_corners=True)

        if not self.merge_sampled_points_w_attn:
            # 5. 整理输出形状
            #    [BT, C, J, N_points] -> [BT, J, N_points, C] -> [BT, J, N_points * C]
            sampled_features = sampled_features.permute(0, 2, 3, 1).reshape(BT, J, self.num_sampling_points * C)
            return sampled_features
        

        # 6. 整理并应用注意力权重 (融合) <--- 新增
        #    attn_logits: [BT, N_points, J, 1] -> [BT, J, N_points]
        attn_logits = attn_logits.squeeze(-1).permute(0, 2, 1)
        #    在 N_points 维度上应用 softmax
        #    attention_weights: [BT, J, N_points]
        attention_weights = F.softmax(attn_logits, dim=-1)
        attention_weights = attention_weights.unsqueeze(1)
        weighted_features = sampled_features * attention_weights
        fused_features = torch.sum(weighted_features, dim=-1)        
        # 8. 整理输出形状
        #    [BT, C, J] -> [BT, J, C]
        fused_features = fused_features.permute(0, 2, 1)        
        
        
        return fused_features


class CrossAttentionFusion(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        # 注意：这里我们假设 skel_channels 和 vision_channels 相同
        # 如果不同，需要先用线性层将它们投影到同一维度
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, skel_feats, vision_feats):
        # skel_feats: [B, C, T, J], vision_feats: [B, C, T, J]
        B, C, T, J = skel_feats.shape
        
        # 展平 T 和 J 维度以适配 Attention
        # [B, C, T, J] -> [B, T*J, C]
        skel_flat = skel_feats.flatten(2).permute(0, 2, 1)
        vision_flat = vision_feats.flatten(2).permute(0, 2, 1)
        
        # 骨架特征是 Query，视觉特征是 Key 和 Value
        # 这意味着 "用骨架的需求去查询视觉信息"
        attn_output, _ = self.attention(
            query=skel_flat, 
            key=vision_flat, 
            value=vision_flat
        )
        
        # 添加残差连接和层归一化
        fused_flat = self.layer_norm(skel_flat + attn_output)
        
        # 恢复原始形状
        # [B, T*J, C] -> [B, C, T, J]
        return fused_flat.permute(0, 2, 1).view(B, C, T, J)
    

class HYBRID_VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq, vision_config, joint_data_type, 
                 deformable_attn_args={
                     'num_heads': 4,
                     'num_samples': 4,
                 }):
        super(HYBRID_VQVAE, self).__init__()

        self.joint_data_type = joint_data_type

        ####################### adjust configs before creating modules #######################
        num_channels_list = vision_config.model.backbone.STAGE4.NUM_CHANNELS
        self.hrnet_output_level = vision_config.model.hybrid.hrnet_output_level   # int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征
        if isinstance(self.hrnet_output_level, int):
            num_channels_list = [num_channels_list[self.hrnet_output_level]]
        elif isinstance(self.hrnet_output_level, list):
            num_channels_list = [num_channels_list[i] for i in self.hrnet_output_level]
        else:
            raise ValueError("hrnet_output_level should be int or list")
        self.num_vision_channels = sum(num_channels_list)

        self.vision_guidance_where = vision_config.model.hybrid.get('vision_guidance_where', 'enc')
        self.vision_guidance_fuse = vision_config.model.hybrid.get('vision_guidance_fuse', 'cat')
        self.vision_guidance_extraLoss = vision_config.model.hybrid.get('vision_guidance_extraLoss', None)
        self.vision_guidance_extraLossConfig = vision_config.model.hybrid.get('vision_guidance_extraLossConfig', None)

        self.vision_guidance_ratio = vision_config.model.hybrid.vision_guidance_ratio
        # assert self.vision_guidance_ratio > 0, "vision_guidance_ratio should be > 0. Use base VQVAE instead of hybrid VQVAE if no vision guidance is needed."
        if self.vision_guidance_ratio == -1:
            code_dim_vision = vq.code_dim
            code_dim_skel = vq.code_dim

            encoder.out_channels = code_dim_skel
            ####################### creating modules #######################
            self.encoder = Encoder(**encoder)
            self.decoder = Decoder(**decoder)
            self.vq = VectorQuantizer(**vq)

            self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)
            self.vision_encoder = VisionEncoder(
                mid_channels=[self.num_vision_channels, 512],
                out_channels=code_dim_vision,
                downsample_time=encoder.downsample_time,
                downsample_joint=[1, 1],
            )


            self.fuse_vis_skel = nn.Linear(code_dim_vision + code_dim_skel, 2)
            self.fuse_vis_skel.weight.data.fill_(0)
            self.fuse_vis_skel.bias.data.fill_(0.5)


        elif self.vision_guidance_ratio == -2:
            raise NotImplementedError
            code_dim_vision = int(vq.code_dim * self.vision_guidance_ratio)
            code_dim_skel = vq.code_dim - code_dim_vision

            encoder.out_channels = code_dim_skel
            ####################### creating modules #######################
            self.encoder = Encoder(**encoder)
            self.decoder = Decoder(**decoder)
            self.vq = VectorQuantizer(**vq)

            self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)
            self.vision_encoder = VisionEncoder(
                mid_channels=[self.num_vision_channels, 512],
                out_channels=code_dim_vision,
                downsample_time=encoder.downsample_time,
                downsample_joint=[1, 1],
            )


            self.num_heads = deformable_attn_args['num_heads']      # 4
            self.num_samples = deformable_attn_args['num_samples']  # 4

            self.coord_embed = nn.Linear(3, code_dim_skel)
            self.norm1 = nn.LayerNorm(code_dim_skel)    # LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            self.attention_weights = nn.Linear(code_dim_skel, self.num_heads * self.num_samples)    # Linear(128,16)
            self.sampling_offsets = nn.Linear(code_dim_skel, 2 * self.num_heads * self.num_samples)

            nn.init.constant_(self.sampling_offsets.weight.data, 0.)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
            for i in range(self.num_samples):
                grid_init[:, i, :] *= i + 1
            with torch.no_grad():
                self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(self.attention_weights.weight.data, 0.)
            nn.init.constant_(self.attention_weights.bias.data, 0.)


        else:
            if self.vision_guidance_fuse == 'ada_sample':
                code_dim_vision = int(vq.code_dim * self.vision_guidance_ratio)
                code_dim_skel = vq.code_dim - code_dim_vision

                encoder.out_channels = code_dim_skel
                ####################### creating modules #######################
                self.encoder = Encoder(**encoder)
                self.decoder = Decoder(**decoder)
                self.vq = VectorQuantizer(**vq)

                if self.vision_guidance_ratio > 0:
                    if vision_config.model.backbone.type == 'qwen2.5vl':
                        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                        model_weight_path = "/data1/wxs/huggingface/Qwen/Qwen2.5-VL-7B-Instruct/"
                        self.vision_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_weight_path,
                                                                                                  trust_remote_code=True,
                                                                                                  torch_dtype=torch.float16, # 使用半精度以节省显存
                                                                                                #   device_map="auto",
                                                                                                  ).visual  # <class 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel'>
                        self.vision_processor = AutoProcessor.from_pretrained(model_weight_path,
                                                                              trust_remote_code=True
                                                                              ).video_processor   # <class 'transformers.models.qwen2_vl.video_processing_qwen2_vl.Qwen2VLVideoProcessor'>
                        
                        self.adaptive_samplers = nn.ModuleList()
                        for channels in [self.vision_backbone.config.hidden_size]: # num_channels_list 是你已有的 [32, 64, 128, 256] 列表
                            self.adaptive_samplers.append(AdaptiveSampler(feature_channels=channels, num_sampling_points=4, merge_sampled_points_w_attn=True))


                        self.vision_encoder = nn.Identity()
                
                    else:
                        self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)

                        self.adaptive_samplers = nn.ModuleList()
                        for channels in num_channels_list: # num_channels_list 是你已有的 [32, 64, 128, 256] 列表
                            self.adaptive_samplers.append(AdaptiveSampler(feature_channels=channels, num_sampling_points=4))

                        # 计算采样后的总特征维度
                        total_sampled_channels = sum([c * 4 for c in num_channels_list]) # 假设 num_sampling_points=4

                        self.vision_encoder = VisionEncoder(
                            mid_channels=[total_sampled_channels, 512],
                            out_channels=code_dim_vision,
                            downsample_time=encoder.downsample_time,
                            downsample_joint=[1, 1],
                        )

            elif self.vision_guidance_fuse == 'cross_attn':
                code_dim_vision = vq.code_dim
                code_dim_skel = vq.code_dim

                encoder.out_channels = code_dim_skel
                ####################### creating modules #######################
                self.encoder = Encoder(**encoder)
                self.decoder = Decoder(**decoder)
                self.vq = VectorQuantizer(**vq)

                self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)
                self.vision_encoder = VisionEncoder(
                    mid_channels=[self.num_vision_channels, 512],
                    out_channels=code_dim_vision,
                    downsample_time=encoder.downsample_time,
                    downsample_joint=[1, 1],
                )

                self.fusion_module = CrossAttentionFusion(channels=code_dim_vision, num_heads=8)

            elif self.vision_guidance_fuse == 'dec_vis_cond':
                code_dim_vision = int(vq.code_dim * self.vision_guidance_ratio)
                code_dim_skel = vq.code_dim - code_dim_vision

                encoder.out_channels = code_dim_skel
                ####################### creating modules #######################
                self.encoder = Encoder(**encoder)
                self.decoder = ConditionalDecoder(cond_dim=code_dim_vision, **decoder)
                self.vq = VectorQuantizer(**vq)

                self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)
                self.vision_encoder = VisionEncoder(
                    mid_channels=[self.num_vision_channels, 512],
                    out_channels=code_dim_vision,
                    downsample_time=encoder.downsample_time,
                    downsample_joint=[1, 1],
                )

            else:
                code_dim_vision = int(vq.code_dim * self.vision_guidance_ratio)
                code_dim_skel = vq.code_dim - code_dim_vision

                encoder.out_channels = code_dim_skel
                ####################### creating modules #######################
                self.encoder = Encoder(**encoder)
                self.decoder = Decoder(**decoder)
                self.vq = VectorQuantizer(**vq)

                if self.vision_guidance_ratio > 0:
                    if vision_config.model.backbone.type == 'qwen2.5vl':
                        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                        model_weight_path = "/data1/wxs/huggingface/Qwen/Qwen2.5-VL-7B-Instruct/"
                        self.vision_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_weight_path,
                                                                                                  trust_remote_code=True,
                                                                                                  torch_dtype=torch.float16, # 使用半精度以节省显存
                                                                                                #   device_map="auto",
                                                                                                  ).visual  # <class 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel'>
                        self.vision_processor = AutoProcessor.from_pretrained(model_weight_path,
                                                                              trust_remote_code=True
                                                                              ).video_processor   # <class 'transformers.models.qwen2_vl.video_processing_qwen2_vl.Qwen2VLVideoProcessor'>
                    elif vision_config.model.backbone.type == 'dinov2':
                        from PIL import Image
                        from transformers import AutoImageProcessor, AutoModel
                        ckpt_path = "/data2/wxs_new/_HUGGINGFACE/dinov2-small"
                        self.vision_processor = AutoImageProcessor.from_pretrained(ckpt_path)
                        self.vision_backbone = AutoModel.from_pretrained(ckpt_path)
                        self.vision_backbone.eval() # 切换到评估模式


                    else:                    
                        self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)



                    self.vision_encoder = VisionEncoder(
                        mid_channels=[self.num_vision_channels, 512],
                        out_channels=code_dim_vision,
                        downsample_time=encoder.downsample_time,
                        downsample_joint=[1, 1],
                    )


    def load_model_weights(self, weight_path):
        safetensors_path = os.path.join(weight_path, "model.safetensors")
        pytorch_bin_path = os.path.join(weight_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            print(f"Loading model from {safetensors_path}")
            state_dict = load_safetensors(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_bin_path):
            print(f"Loading model from {pytorch_bin_path}")
            state_dict = torch.load(pytorch_bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {weight_path}")
        self.load_state_dict(state_dict, strict=True)
        return


    
    def to(self, device):
        return super().to(device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.vq = self.vq.to(device)
        if self.vision_guidance_ratio == -1:
            self.vision_backbone = self.vision_backbone.to(device)
            self.vision_encoder = self.vision_encoder.to(device)
            self.fuse_vis_skel = self.fuse_vis_skel.to(device)
        elif self.vision_guidance_ratio == -2:
            self.vision_backbone = self.vision_backbone.to(device)
            self.vision_encoder = self.vision_encoder.to(device)
            self.fuse_vis_skel = self.fuse_vis_skel.to(device)
            self.coord_embed = self.coord_embed.to(device)
            self.norm1 = self.norm1.to(device)
            self.attention_weights = self.attention_weights.to(device)
            self.sampling_offsets = self.sampling_offsets.to(device)
        elif self.vision_guidance_ratio > 0:
            self.vision_backbone = self.vision_backbone.to(device)
            self.vision_encoder = self.vision_encoder.to(device)
        self.device = device
        return self
        
    

    def motion_weighted_downsample(self, skeleton_data: torch.Tensor, motion_scale: float = 0.1) -> torch.Tensor:
        """
        使用运动加权平均将骨架数据从 [B, T, V, C] 降采样到 [B, T/2, V, C]。

        Args:
            skeleton_data (torch.Tensor): 
                输入的骨架数据，形状为 [B, T, V, C]。
                例如 [4, 64, 17, 2]。
            
            motion_scale (float): 
                一个缩放因子，用于调整运动幅度对 sigmoid 的敏感度。
                较小的值 (如 0.1) 使其对较大的运动（例如 10 像素）
                才产生高权重 (alpha 接近 1.0)。

        Returns:
            torch.Tensor: 
                降采样后的骨架数据，形状为 [B, T/2, V, C]。
                例如 [4, 32, 17, 2]。
        """
        
        # 1. 确保时间维度 T 是偶数
        # 如果 T 是奇数，丢弃最后一帧
        if skeleton_data.shape[1] % 2 != 0:
            # [B, T, V, C] -> [B, T-1, V, C]
            skeleton_data = skeleton_data[:, :-1, :, :]
            
        # 2. 获取批次和形状信息
        # B = Batch Size, T = Time (now even)
        # V = Vertices/Joints (17), C = Coordinates (2)
        B, T, V, C = skeleton_data.shape
        T_half = T // 2
        
        # 3. 重塑数据以配对 (pose_a, pose_b)
        # [B, T, V, C] -> [B, T/2, 2, V, C]
        paired_data = skeleton_data.view(B, T_half, 2, V, C)
        
        # 4. 分离出 pose_a 和 pose_b
        # 形状 [B, T/2, V, C]
        pose_a = paired_data[:, :, 0]
        pose_b = paired_data[:, :, 1]
        
        # 5. 计算两帧之间的运动 (欧氏距离)
        # [B, T/2, V, C]
        motion_diff = pose_b - pose_a
        
        # 6. 计算每个关节点的运动幅度
        # dim=-1 (即 C 维度) 是 (x, y) 坐标
        # torch.norm 在此计算 sqrt(x_diff^2 + y_diff^2)
        # 形状: [B, T/2, V]
        motion_magnitude = torch.norm(motion_diff, dim=-1)
        
        # 7. 计算自适应权重 (alpha)
        # 我们使用 sigmoid 函数，它具有以下特性：
        # - motion = 0   -> sigmoid(0) = 0.5 (低运动 -> 平均)
        # - motion = high -> sigmoid(high) = 1.0 (高运动 -> 取 pose_b)
        #
        # 形状: [B, T/2, V] -> [B, T/2, V, 1]
        alpha_per_joint = torch.sigmoid(motion_magnitude * motion_scale).unsqueeze(-1)
        
        # 8. 应用加权平均
        # new_pose = (1 - alpha) * pose_a + alpha * pose_b
        # [B, T/2, V, 1] 会自动广播到 [B, T/2, V, C]
        new_pose = (1.0 - alpha_per_joint) * pose_a + alpha_per_joint * pose_b
        
        return new_pose


    def get_vision_feats(self, joint3d_video, video_rgb):  # [B,T,H,W,3]
        with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
            assert self.vision_guidance_ratio > 0 or self.vision_guidance_ratio == -1 or self.vision_guidance_ratio == -2
            B, T = joint3d_video.shape[:2]
            if self.vision_backbone.__class__.__name__ == 'Qwen2_5_VisionTransformerPretrainedModel':
                assert len(video_rgb) == B, "video_rgb should be a list of length B, each item is a dict returned by Qwen2.5-VL processor."
                VIDEO_FEATURE_LIST = []
                for video_rgb_per_video in video_rgb:
                    pixel_values_videos = video_rgb_per_video['pixel_values_videos'].type(self.vision_backbone.dtype).to(self.vision_backbone.device)
                    video_grid_thw = video_rgb_per_video['video_grid_thw']
                    
                    ################################## self.visual ##################################
                    # video_embeds = self.vision_backbone(pixel_values_videos,
                    #                                     grid_thw=video_grid_thw,
                    #                                     )
                    hidden_states = pixel_values_videos     # [64512,1176]
                    grid_thw = video_grid_thw
                    hidden_states = self.vision_backbone.patch_embed(hidden_states) # torch.Size([64512, 1280])
                    rotary_pos_emb = self.vision_backbone.rot_pos_emb(grid_thw)
                    window_index, cu_window_seqlens = self.vision_backbone.get_window_index(grid_thw)   # [16128,]
                    cu_window_seqlens = torch.tensor(
                        cu_window_seqlens,
                        device=hidden_states.device,
                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                    )
                    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

                    seq_len, _ = hidden_states.size()
                    hidden_states = hidden_states.reshape(seq_len // self.vision_backbone.spatial_merge_unit, self.vision_backbone.spatial_merge_unit, -1)  # [64512,1280] -> [16128,4,1280]
                    hidden_states = hidden_states[window_index, :, :]
                    hidden_states = hidden_states.reshape(seq_len, -1)  # [16128,4,1280] -> [64512,1280]
                    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.vision_backbone.spatial_merge_unit, self.vision_backbone.spatial_merge_unit, -1)
                    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
                    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                    position_embeddings = (emb.cos(), emb.sin())

                    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                        dim=0,
                        # Select dtype based on the following factors:
                        #  - FA2 requires that cu_seqlens_q must have dtype int32
                        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                        # See https://github.com/huggingface/transformers/pull/34852 for more information
                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                    )
                    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

                    for layer_num, blk in enumerate(self.vision_backbone.blocks):
                        if layer_num in self.vision_backbone.fullatt_block_indexes:
                            cu_seqlens_now = cu_seqlens
                        else:
                            cu_seqlens_now = cu_window_seqlens

                        hidden_states = blk(
                            hidden_states,
                            cu_seqlens=cu_seqlens_now,
                            position_embeddings=position_embeddings,
                        )
                    # torch.Size([64512, 1280])

                    hidden_states = self.vision_backbone.merger.ln_q(hidden_states) # [64512,1280] -> [64512,1280]


                    block_output_shuffled = hidden_states

                    # 1. 获取反打乱索引
                    reverse_indices = torch.argsort(window_index)
                    
                    # 2. 获取参数
                    spatial_merge_unit = self.vision_backbone.spatial_merge_unit # 应该是 4
                    seq_len, hidden_dim = block_output_shuffled.shape # 64512, 1280
                    num_groups = seq_len // spatial_merge_unit          # 16128
                    
                    # 3. 将乱序的 token 重塑为乱序的 组
                    # [64512, 1280] -> [16128, 4, 1280]
                    shuffled_groups = block_output_shuffled.view(num_groups, spatial_merge_unit, hidden_dim)
                    
                    # 4. 使用 reverse_indices 将 组 恢复到正确顺序
                    # [16128, 4, 1280] -> [16128, 4, 1280] (顺序已恢复)
                    unshuffled_groups = shuffled_groups[reverse_indices, :, :]
                    
                    # 5. 将顺序正确的 组 摊平为 顺序正确的 token 序列
                    # [16128, 4, 1280] -> [64512, 1280]
                    pure_visual_features_ordered = unshuffled_groups.view(seq_len, hidden_dim)  # [64512,1280]
                    
                    # --- 结束 ---

                    # pure_visual_features_ordered 就是你想要的特征！
                    # 你可以将其用于下游任务。
                    # your_downstream_model(pure_visual_features_ordered)
                    


                    """
                    # 原始代码继续执行 (注意：它是在乱序的 hidden_states 上操作)
                    # hidden_states = self.vision_backbone.merger(hidden_states)  # [64512,1280] -> [16128,3584]
                    hidden_states = self.vision_backbone.merger.ln_q(hidden_states) # [64512,1280] -> [64512,1280]
                    hidden_states = hidden_states.view(-1, self.vision_backbone.merger.hidden_size) # [64512,1280] -> [16128,5120]
                    hidden_states = self.vision_backbone.merger.mlp(hidden_states) # [16128,5120] -> [16128,3584]

                    # 原始的 reverse_indices 操作是在 merger 之后进行的
                    # reverse_indices = torch.argsort(window_index)
                    hidden_states = hidden_states[reverse_indices, :]   # torch.Size([16128, 3584])
                    """
                    ################################## self.visual ##################################
                    # split_sizes = (video_grid_thw.prod(-1) // self.vision_backbone.spatial_merge_size**2).tolist()
                    # video_embeds = torch.split(video_embeds, split_sizes)

                    
                    assert pure_visual_features_ordered.shape[0] == video_grid_thw.prod(-1)
                    pure_visual_features_ordered = pure_visual_features_ordered.reshape(*video_grid_thw[0].tolist(), pure_visual_features_ordered.shape[-1])
                    # [2016,1280] -> [T'=8,18,14,1280]
                    pure_visual_features_ordered = pure_visual_features_ordered.permute(0, 3, 1, 2).contiguous()  # [T'=8,1280,18,14]
                    VIDEO_FEATURE_LIST.append(pure_visual_features_ordered)

                # VIDEO_FEATURE_LIST = torch.stack(VIDEO_FEATURE_LIST, dim=0)  # [B, T', 1280, H', W']
                VIDEO_FEATURE_LIST = torch.cat(VIDEO_FEATURE_LIST, dim=0)  # [BT', 1280, H', W']


                image_feature_list = [VIDEO_FEATURE_LIST]


                joint3d_video_Tmerged = self.motion_weighted_downsample(joint3d_video)  # [B,T,17,3] -> [B,T',17,3]
                T_merged = joint3d_video_Tmerged.shape[1]


                T = T_merged


                assert VIDEO_FEATURE_LIST.shape[0] == B * T_merged, "The number of video features should match the number of joint3d frames after T downsampling."

                joint3d_images = joint3d_video_Tmerged.reshape(-1, *joint3d_video.shape[2:])    # [B*T,17,3]
                grid_sample_ref = joint3d_images[..., :2].unsqueeze(-2)  # [B*T,17,1,2]




            else:
                video_rgb = video_rgb.permute(0, 1, 4, 2, 3).contiguous()  # [B,T,3,H,W]
                images = video_rgb.reshape(-1, *video_rgb.shape[2:])  # [B*T,3,H,W]
                
                image_feature_list = self.vision_backbone(images)
                # [[BT,32,64,48], [BT,64,32,24], [BT,128,16,12], [BT,256,8,6]]

                if isinstance(self.hrnet_output_level, int):
                    image_feature_list = [image_feature_list[self.hrnet_output_level]]
                elif isinstance(self.hrnet_output_level, list):
                    image_feature_list = [image_feature_list[i] for i in self.hrnet_output_level]


                joint3d_images = joint3d_video.reshape(-1, *joint3d_video.shape[2:])    # [B*T,17,3]
                grid_sample_ref = joint3d_images[..., :2].unsqueeze(-2)  # [B*T,17,1,2]


            if self.vision_guidance_ratio == -2:
                # [BT,17,3] -> [BT,17,code_dim_skel]
                joint_embed = self.coord_embed(joint3d_images)
                joint_embed = self.norm1(joint_embed)

                # [BT,17,code_dim_skel] -> [BT,17,num_heads,num_samples]
                attn_weights = self.attention_weights(joint_embed).view(-1, 17, self.num_heads, self.num_samples)
                attn_weights = F.softmax(attn_weights, -1) # [BT,17,num_heads,num_samples]

                # [BT,17,code_dim_skel] -> [BT,17,num_heads*num_samples,2]
                offsets = self.sampling_offsets(joint_embed).reshape(-1, 17, self.num_heads * self.num_samples, 2).tanh()
                
                # grid_sample_ref: [BT,17,1,2], offsets: [BT,17,num_heads*num_samples,2]
                # pos: [BT,17,num_heads*num_samples,2]
                pos = grid_sample_ref + offsets

                features_ref_list = []
                for features in image_feature_list: # features: [BT,C,H,W]
                    # sampled_feats: [BT,C,17,num_heads*num_samples]
                    sampled_feats = F.grid_sample(features, pos, align_corners=True)
                    
                    C = sampled_feats.shape[1]
                    # -> [BT,C,17,num_heads,num_samples]
                    sampled_feats = sampled_feats.view(-1, C, 17, self.num_heads, self.num_samples)
                    
                    # attn_weights: [BT,17,num_heads,num_samples] -> [BT,1,17,num_heads,num_samples]
                    # weighted_sum: [BT,C,17,num_heads]
                    weighted_sum = (sampled_feats * attn_weights.unsqueeze(1)).sum(dim=-1)
                    
                    # -> [BT,C,17]
                    features_ref = weighted_sum.sum(dim=-1)
                    
                    # -> [BT,17,C]
                    features_ref = features_ref.permute(0, 2, 1).contiguous()
                    features_ref_list.append(features_ref)
                
                video_ref_features = torch.cat(features_ref_list, dim=-1)
                video_ref_features = video_ref_features.reshape(B, T, *video_ref_features.shape[1:])
                return video_ref_features.permute(0, 3, 1, 2).contiguous()
                
            
            if self.vision_guidance_fuse == 'ada_sample':
                grid_sample_ref_coords = joint3d_images[..., :2]
                features_ref_list = []
                # 使用 enumerate 来同时获取特征和对应的 sampler
                for i, features in enumerate(image_feature_list):
                    # 调用对应的 AdaptiveSampler
                    # 输入: features [BT, C, H, W], coords [BT, J, 2]
                    # 输出: features_ref [BT, J, C * N_points]
                    features_ref = self.adaptive_samplers[i](features, grid_sample_ref_coords)
                    features_ref_list.append(features_ref)            
                # features_ref_list: [[BT,17,32*4], [BT,17,64*4], [BT,17,128*4], [BT,17,256*4]]
                video_ref_features = torch.cat(features_ref_list, dim=-1)
                # video_ref_features: [BT, 17, 480*4=1920]
                
                video_ref_features = video_ref_features.reshape(B, T, *video_ref_features.shape[1:])
                # [B, T, 17, 1920]

                # 调整为 [B, C, T, J] 以适配你的 VisionEncoder


                if not hasattr(self, f'ada_sample_successfully_applied'):
                    print("Adaptive sampling applied successfully.")
                    setattr(self, f'ada_sample_successfully_applied', True)
                return video_ref_features.permute(0, 3, 1, 2).contiguous()




            features_ref_list = []
            for features in image_feature_list:
                features_ref = F.grid_sample(features, grid_sample_ref, align_corners=True)
                # features: [BT,256,8,6]; grid_sample_ref: [BT,17,1,2] >>> features_ref: [BT,256,17,1]
                # TODO: 如果 grid_sample_ref 的倒数第二维不是1, 而是3 (比如人体关节的特征不是一个点, 而是一个小区域), 会怎么样?
                features_ref = features_ref.squeeze(-1).permute(0, 2, 1).contiguous()   # [BT,17,256]
                features_ref_list.append(features_ref)
            # features_ref_list: [[BT,17,32], [BT,17,64], [BT,17,128], [BT,17,256]]
            video_ref_features = torch.cat(features_ref_list, dim=-1)  # [BT,17,32+64+128+256=480]
            video_ref_features = video_ref_features.reshape(B, T, *video_ref_features.shape[1:])  # [B,T,17,480]

            return video_ref_features.permute(0, 3, 1, 2).contiguous()  # [B,480,T,17]


    def forward(self, batch_dict, return_vq=False):
        joint_gt = batch_dict[self.joint_data_type].clone()
        joint3d_video = batch_dict[self.joint_data_type]     # [B,T,17,3]

        if self.vision_guidance_ratio > 0 or self.vision_guidance_ratio == -1 or self.vision_guidance_ratio == -2:
            video_rgb = batch_dict.video_rgb  # [B,T,H,W,3]
            if not isinstance(video_rgb, torch.Tensor):
                VIDEO_BATCH = []
                for batch_id in range(len(video_rgb)):
                    img_paths = video_rgb[batch_id]  # (T,)
                    VIDEO = []
                    for img_id in range(img_paths.shape[0]):
                        img = Image.open(img_paths[img_id])
                        if img.mode != "RGB":
                            img = img.convert('RGB')
                        # VIDEO_BATCH.append(img)
                        VIDEO.append(img)
                    # video_dict = self.vision_processor(VIDEO)['pixel_values_videos']
                    video_dict = self.vision_processor(VIDEO)
                    VIDEO_BATCH.append(video_dict)
                # VIDEO_BATCH = self.vision_processor(VIDEO_BATCH)
                # VIDEO_BATCH = torch.stack(VIDEO_BATCH, dim=0)
                video_rgb = VIDEO_BATCH
            vision_feats = self.get_vision_feats(joint3d_video, video_rgb)
        else:
            vision_feats = None

        joint_feats = joint3d_video.permute(0, 3, 1, 2)   # [B,49,17,3] -> [B,3,49,17]
        indices = None
        if not self.vq.is_train:
            joint_feats, loss, indices, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=min(8, joint_gt.shape[1]), encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
        else:
            tuple_return = self.encdec_slice_frames(joint_feats, frame_batch_size=min(8, joint_gt.shape[1]), encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
            if self.vision_guidance_extraLoss == 'infonce':
                joint_feats, loss, perplexity, _, SKEL_VIS_DICT = tuple_return
                B = SKEL_VIS_DICT['skel'].shape[0]
                
                joint_vec = F.adaptive_avg_pool2d(SKEL_VIS_DICT['skel'], 1).squeeze().view(B, -1) # [B, C]
                vision_vec = F.adaptive_avg_pool2d(SKEL_VIS_DICT['vis'], 1).squeeze().view(B, -1) # [B, C]
                
                # 计算对比损失
                # 需要对特征进行归一化
                joint_vec = F.normalize(joint_vec, p=2, dim=1)
                vision_vec = F.normalize(vision_vec, p=2, dim=1)
                
                # 计算 logits
                logits_per_joint = joint_vec @ vision_vec.t() / self.vision_guidance_extraLossConfig['infonce']['temperature']
                logits_per_vision = vision_vec @ joint_vec.t() / self.vision_guidance_extraLossConfig['infonce']['temperature']
                
                # 对称的交叉熵损失
                labels = torch.arange(B, device=joint_vec.device)
                loss_j = F.cross_entropy(logits_per_joint, labels)
                loss_v = F.cross_entropy(logits_per_vision, labels)
                contrastive_loss = (loss_j + loss_v) / 2
                
                # 将 contrastive_loss 加入到你的总损失中
                extra_loss = self.vision_guidance_extraLossConfig['infonce']['loss_weight'] * contrastive_loss
            else:
                joint_feats, loss, perplexity, _ = tuple_return
        if return_vq:
            return joint_feats, loss
        joint_feats, _, _, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=min(2, joint_feats.shape[2]), encdec=self.decoder, return_vq=return_vq, vision_feats=vision_feats)
        joint_feats = joint_feats.permute(0, 2, 3, 1)
        if self.vq.is_train:
            if self.vision_guidance_extraLoss is not None:
                return joint_feats, loss, perplexity, joint_gt, extra_loss
            return joint_feats, loss, perplexity, joint_gt
        return joint_feats, loss, indices, joint_gt

    def encdec_slice_frames(self, joint_feats, frame_batch_size, encdec, return_vq, vision_feats=None):
        with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
            num_frames = joint_feats.shape[2]
            remaining_frames = num_frames % frame_batch_size
            joint_output = []
            loss_output = []
            perplexity_output = []

            if encdec == self.encoder and self.vision_guidance_extraLoss == 'infonce':
                SKEL_VIS_DICT = defaultdict(list)

            for i in range(num_frames // frame_batch_size):
                remaining_frames = num_frames % frame_batch_size
                start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
                end_frame = frame_batch_size * (i + 1) + remaining_frames
                joint_feats_intermediate = joint_feats[:, :, start_frame:end_frame] # [B,3,8,17]

                if encdec == self.encoder:
                    if vision_feats.shape[2] == joint_feats.shape[2]:
                        vision_start_frame = start_frame
                        vision_end_frame = end_frame
                    elif vision_feats.shape[2] * 2 == joint_feats.shape[2]:
                        vision_start_frame = start_frame // 2
                        vision_end_frame = end_frame // 2
                    else:
                        print("joint_feats.shape[2]=", joint_feats.shape[2], "vision_feats.shape[2]=", vision_feats.shape[2])
                        raise NotImplementedError


                if encdec == self.decoder and vision_feats is not None and self.vision_guidance_fuse == 'dec_vis_cond':
                    vision_feats_intermediate = vision_feats[:, :, vision_start_frame:vision_end_frame] # [B,480,8,17]
                    vision_encoded = self.vision_encoder(vision_feats_intermediate) # [B,code_dim_vision,2,17]
                    vision_condition_vector = F.adaptive_avg_pool2d(vision_encoded, (1, 1)).squeeze(-1).squeeze(-1)
                    joint_feats_intermediate = encdec(joint_feats_intermediate, vision_condition_vector) # Enc:[B,3072,3,17]
                else:
                    joint_feats_intermediate = encdec(joint_feats_intermediate) # Enc:[B,3072,3,17]

                if encdec == self.encoder and vision_feats is not None:
                    vision_feats_intermediate = vision_feats[:, :, vision_start_frame:vision_end_frame] # [B,480,8,17]
                    vision_encoded = self.vision_encoder(vision_feats_intermediate) # [B,code_dim_vision,2,17]


                    if self.vision_guidance_extraLoss == 'infonce':
                        SKEL_VIS_DICT['skel'].append(joint_feats_intermediate)
                        SKEL_VIS_DICT['vis'].append(vision_encoded)


                    if self.vision_guidance_fuse == 'cross_attn':
                        joint_feats_intermediate = self.fusion_module(joint_feats_intermediate, vision_encoded)

                    else:
                        feats_list = [joint_feats_intermediate, vision_encoded]
                        joint_feats_intermediate = torch.cat(feats_list, dim=1) # [B,code_dim,2,17]

                        if self.vision_guidance_ratio == -1:
                            fuse_weights = self.fuse_vis_skel(joint_feats_intermediate.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                            fuse_weights = fuse_weights.softmax(dim=1)
                            feats_tmp = 0
                            for i in range(len(feats_list)):
                                feats_tmp += fuse_weights[:, i:i + 1, ...] * feats_list[i]
                            joint_feats_intermediate = feats_tmp
                        


                joint_output.append(joint_feats_intermediate)

            joint_concat = torch.cat(joint_output, dim=2)


            if encdec == self.encoder and self.vision_guidance_extraLoss == 'infonce':
                for k, v in SKEL_VIS_DICT.items():
                    SKEL_VIS_DICT[k] = torch.cat(v, dim=2)


            if encdec == self.encoder and self.vq is not None and not self.vq.is_train:
                joint_output, loss, indices = self.vq(joint_concat, return_vq=return_vq)
                tuple_return = (joint_output, loss, indices, joint_concat.shape)
                return tuple_return
            elif encdec == self.encoder and self.vq is not None and self.vq.is_train:
                joint_output, loss, preplexity = self.vq(joint_concat)
                tuple_return = (joint_output, loss, preplexity, joint_concat.shape)
                if self.vision_guidance_extraLoss == 'infonce':
                    tuple_return = tuple_return + (SKEL_VIS_DICT,)
                return tuple_return
            else:
                return joint_concat, None, None, joint_concat.shape

    ############### only encode for inference ###############
    def encode(self, joint3d_video, video_rgb=None, return_vq=False):
        assert not self.vq.is_train, "Only support encode when vq is not training."
        if video_rgb is not None:
            vision_feats = self.get_vision_feats(joint3d_video, video_rgb)
        else:
            vision_feats = None
        joint_feats = joint3d_video.permute(0, 3, 1, 2)   # [B,49,17,3] -> [B,3,49,17]
        _, _, indices, quant_shape = self.encdec_slice_frames(joint_feats, frame_batch_size=min(8, joint_feats.shape[-2]), encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
        return indices, quant_shape

    def get_code_from_indices(self, indices):
        """
        [NEW HELPER METHOD]
        Converts a tensor of indices into the corresponding codebook vectors.
        This is the core "dequantization" step.
        
        Args:
            indices (torch.Tensor): A tensor of indices, shape [B, T_quant, J_quant].
        
        Returns:
            torch.Tensor: The corresponding codebook vectors, shape [B, C, T_quant, J_quant].
        """
        if not hasattr(self, 'vq'):
            raise ValueError("VectorQuantizer (self.vq) is not available.")
        
        # Flatten the indices tensor for embedding lookup
        # Shape: [B, T_quant, J_quant] -> [B * T_quant * J_quant]
        flat_indices = indices.view(-1)
        
        # Use the VQ's dequantize method (which is essentially F.embedding)
        # Shape: [B * T_quant * J_quant] -> [B * T_quant * J_quant, C]
        dequantized_vectors = self.vq.dequantize(flat_indices)
        
        # Reshape the vectors back to the 4D tensor format expected by the decoder
        batch_size, t_quant, j_quant = indices.shape
        code_dim = dequantized_vectors.shape[-1]
        
        # Shape: [B * T_quant * J_quant, C] -> [B, T_quant, J_quant, C]
        vectors_reshaped = dequantized_vectors.view(batch_size, t_quant, j_quant, code_dim)
        
        # Permute to the [B, C, T, J] format for convolutional layers
        # Shape: [B, T_quant, J_quant, C] -> [B, C, T_quant, J_quant]
        return vectors_reshaped.permute(0, 3, 1, 2).contiguous()

    def decode(self, indices: torch.Tensor):
        if self.vision_guidance_where == 'encdec':
            raise NotImplementedError
        """
        [REWRITTEN METHOD]
        Decodes a batch of indices into the reconstructed skeleton data.
        
        Args:
            indices (torch.Tensor): The code indices to decode, with shape [B, T_quant, J_quant].
        
        Returns:
            torch.Tensor: The reconstructed skeleton data, with shape [B, T, J, C].
        """
        # 1. Convert indices to codebook vectors
        # Input: [B, T_quant, J_quant], Output: [B, C, T_quant, J_quant]
        quantized_vectors = self.get_code_from_indices(indices)
        
        # 2. Decode the vectors using the decoder network
        # The `encdec_slice_frames` is used here to handle potentially long sequences.
        # Input: [B, C, T_quant, J_quant], Output: [B, C, T, J]
        reconstructed_x, _, _, _ = self.encdec_slice_frames(
            quantized_vectors, 
            frame_batch_size=min(2, quantized_vectors.shape[2]),  # This can be adjusted based on memory
            encdec=self.decoder, 
            return_vq=False
        )
        
        # 3. Permute the output to the standard [B, T, J, C] format
        # Input: [B, C, T, J], Output: [B, T, J, C]
        return reconstructed_x.permute(0, 2, 3, 1).contiguous()

    def decode_from_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        直接从量化后的嵌入向量解码出3D姿态。
        Args:
            quantized (torch.Tensor): 量化后的嵌入向量, 形状为 (B, T, J, D)
        Returns:
            torch.Tensor: 解码后的3D姿态, 形状为 (B, T_out, J_out, 3)
        """
        b, t, j, d = quantized.shape
        # 解码器输入需要 (B, D, T, J)
        quantized_reshaped = quantized.permute(0, 3, 1, 2).contiguous()
        out = self.decoder(quantized_reshaped)
        return out

    def decode_with_optimization(
        self, 
        indices: torch.Tensor, 
        known_xy: torch.Tensor, 
        num_steps: int = 50, 
        lr: float = 0.01
    ) -> torch.Tensor:
        """
        [NEW METHOD - STRATEGY 2]
        Decodes indices by optimizing the latent vectors to match known 2D joints.

        Args:
            indices (torch.Tensor): Initial code indices, shape [B, T_quant, J_quant].
            known_xy (torch.Tensor): Ground truth 2D joints, shape [B, T, J, 2].
            num_steps (int): Number of optimization steps.
            lr (float): Learning rate for the optimization.

        Returns:
            torch.Tensor: The final, optimized 3D pose, shape [B, T, J, 3].
        """
        if not self.training:
            self.eval() # Ensure model is in eval mode

        # 1. Get initial latent vectors from indices
        # Output shape: [B, C, T_quant, J_quant]
        quantized_vectors = self.get_code_from_indices(indices)
        
        # Permute to [B, T_quant, J_quant, C] to make it easier to work with
        quantized_vectors = quantized_vectors.permute(0, 2, 3, 1)
        
        # Make the latent vectors trainable for optimization
        quantized_vectors.requires_grad_(True)

        # 2. Set up the optimizer
        optimizer = torch.optim.Adam([quantized_vectors], lr=lr)

        print(f"Starting latent space optimization for {num_steps} steps...")

        # 3. Optimization loop
        for i in range(num_steps):
            optimizer.zero_grad()
            
            # Decode the current latent vectors into a 3D pose
            # Note: decode_from_quantized returns [B, 3, T_out, J_out]
            reconstructed_pose_raw = self.decode_from_quantized(quantized_vectors)
            # Permute to [B, T_out, J_out, 3]
            reconstructed_pose = reconstructed_pose_raw.permute(0, 2, 3, 1)
            
            # Handle potential shape mismatch due to decoder upsampling
            if reconstructed_pose.shape[1] != known_xy.shape[1]:
                target_xy = F.interpolate(
                    known_xy.permute(0, 3, 1, 2), # B, 2, T, J
                    size=(reconstructed_pose.shape[1], reconstructed_pose.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1) # B, T_out, J_out, 2
            else:
                target_xy = known_xy

            # Get the XY part of the reconstruction
            reconstructed_xy = reconstructed_pose[..., :2]
            
            # 4. Calculate loss against the known XY
            loss = F.mse_loss(reconstructed_xy, target_xy)
            
            # 5. Backpropagate and update the latent vectors
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"  Step [{i+1}/{num_steps}], Loss: {loss.item():.6f}")

        # 6. Get the final result from the optimized latent vectors
        print("Optimization finished.")
        final_reconstructed_pose_raw = self.decode_from_quantized(quantized_vectors.detach())
        final_reconstructed_pose = final_reconstructed_pose_raw.permute(0, 2, 3, 1)

        return final_reconstructed_pose.contiguous()