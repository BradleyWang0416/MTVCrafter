import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .vqvae import SMPL_VQVAE, ResBlock, Downsample, Encoder, Decoder, VectorQuantizer

import sys
sys.path.append('/home/wxs/ContextAware-PoseFormer/ContextPose/mvn/models/')
import pose_hrnet
sys.path.remove('/home/wxs/ContextAware-PoseFormer/ContextPose/mvn/models/')

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
    

class HYBRID_VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq, vision_config, joint_data_type):
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

        self.vision_guidance_ratio = vision_config.model.hybrid.vision_guidance_ratio
        # assert self.vision_guidance_ratio > 0, "vision_guidance_ratio should be > 0. Use base VQVAE instead of hybrid VQVAE if no vision guidance is needed."
        code_dim_vision = int(vq.code_dim * self.vision_guidance_ratio)
        code_dim_skel = vq.code_dim - code_dim_vision

        encoder.out_channels = code_dim_skel
        ####################### creating modules #######################
        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)
        self.vq = VectorQuantizer(**vq)

        if self.vision_guidance_ratio > 0:
            self.vision_backbone = pose_hrnet.get_pose_net(vision_config.model.backbone)
            self.vision_encoder = VisionEncoder(
                mid_channels=[self.num_vision_channels, 512],
                out_channels=code_dim_vision,
                downsample_time=[2, 2],
                downsample_joint=[1, 1],
            )
    
    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.vq = self.vq.to(device)
        if self.vision_guidance_ratio > 0:
            self.vision_backbone = self.vision_backbone.to(device)
            self.vision_encoder = self.vision_encoder.to(device)
        self.device = device
        return self
        
    

    def get_vision_feats(self, joint3d_video, video_rgb):  # [B,T,H,W,3]
        assert self.vision_guidance_ratio > 0
        B, T = joint3d_video.shape[:2]
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

        if self.vision_guidance_ratio > 0:
            video_rgb = batch_dict.video_rgb  # [B,T,H,W,3]
            vision_feats = self.get_vision_feats(joint3d_video, video_rgb)
        else:
            vision_feats = None

        joint_feats = joint3d_video.permute(0, 3, 1, 2)   # [B,49,17,3] -> [B,3,49,17]
        indices = None
        if not self.vq.is_train:
            joint_feats, loss, indices, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=8, encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
        else:
            joint_feats, loss, perplexity, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=8, encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
        if return_vq:
            return joint_feats, loss
        joint_feats, _, _, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=2, encdec=self.decoder, return_vq=return_vq)
        joint_feats = joint_feats.permute(0, 2, 3, 1)
        if self.vq.is_train:
            return joint_feats, loss, perplexity, joint_gt
        return joint_feats, loss, indices  

    def encdec_slice_frames(self, joint_feats, frame_batch_size, encdec, return_vq, vision_feats=None):
        num_frames = joint_feats.shape[2]
        remaining_frames = num_frames % frame_batch_size
        joint_output = []
        loss_output = []
        perplexity_output = []
        for i in range(num_frames // frame_batch_size):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            joint_feats_intermediate = joint_feats[:, :, start_frame:end_frame] # [B,3,8,17]
            joint_feats_intermediate = encdec(joint_feats_intermediate) # Enc:[B,3072,3,17]

            if encdec == self.encoder and vision_feats is not None:
                vision_feats_intermediate = vision_feats[:, :, start_frame:end_frame] # [B,480,8,17]
                vision_encoded = self.vision_encoder(vision_feats_intermediate) # [B,code_dim_vision,2,17]
                joint_feats_intermediate = torch.cat([joint_feats_intermediate, vision_encoded], dim=1) # [B,code_dim,2,17]

            joint_output.append(joint_feats_intermediate)

        joint_concat = torch.cat(joint_output, dim=2)

        if encdec == self.encoder and self.vq is not None and not self.vq.is_train:
            joint_output, loss, indices = self.vq(joint_concat, return_vq=return_vq)
            return joint_output, loss, indices, joint_concat.shape
        elif encdec == self.encoder and self.vq is not None and self.vq.is_train:
            joint_output, loss, preplexity = self.vq(joint_concat)
            return joint_output, loss, preplexity, joint_concat.shape
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
        _, _, indices, quant_shape = self.encdec_slice_frames(joint_feats, frame_batch_size=8, encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
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
            frame_batch_size=2,  # This can be adjusted based on memory
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