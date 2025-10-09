import torch
import torch.nn as nn
from .vqvae import Upsample

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 group_num=32,
                 max_channels=512):
        super(ResBlock, self).__init__()
        # Note: The original padding logic seems unusual. 
        # A standard ResBlock would use padding=1, dilation=1.
        # I will keep your original logic.
        skip = max(1, max_channels // out_channels - 1) 
        self.block = nn.Sequential(
            nn.GroupNorm(group_num, in_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=skip, dilation=skip),
            nn.GroupNorm(group_num, out_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.conv_short = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        hidden_states = self.block(x)
      # The original shortcut logic was slightly incorrect for shape changes. Corrected here.
        x_res = self.conv_short(x)
        return x_res + hidden_states

class AdaGN(nn.Module):
    """
    自适应组归一化 (Adaptive Group Normalization)。
    根据输入的条件向量 cond，生成缩放(scale)和偏移(shift)参数。
    """
    def __init__(self, num_groups, num_channels, cond_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=False)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_channels * 2),
        )

    def forward(self, x, cond):
        # x shape: [B, C, H, W]
        # cond shape: [B, cond_dim]
        
        # 归一化输入特征
        x_normalized = self.group_norm(x)
        
        # 从条件向量生成 scale 和 shift
        style = self.cond_proj(cond) # shape: [B, C*2]
        scale, shift = torch.chunk(style, 2, dim=1) # shape: [B, C], [B, C]
        
        # 将 scale 和 shift 调整为 [B, C, 1, 1] 以进行广播
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        # 应用缩放和偏移，使用 (1 + scale) 以保证初始状态是单位变换，有助于训练稳定
        return x_normalized * (1 + scale) + shift

class ConditionalResBlock(nn.Module):
    """
    带有条件引导的残差块，适配你原有的 ResBlock 结构。
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 cond_dim,
                 group_num=32,
                 max_channels=512):
        super().__init__()
        skip = max(1, max_channels // out_channels - 1)
        
        self.norm1 = AdaGN(group_num, in_channels, cond_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=skip, dilation=skip)
        
        self.norm2 = AdaGN(group_num, out_channels, cond_dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.conv_short = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, cond):
        # x shape: [B, C, H, W]
        # cond shape: [B, cond_dim]
        
        residual = self.conv_short(x)
        
        x = self.norm1(x, cond)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.norm2(x, cond)
        x = self.act2(x)
        x = self.conv2(x)
        
        return x + residual

class ConditionalDecoder(nn.Module):
    """
    修改后的解码器，接受一个额外的视觉条件向量。
    """
    def __init__(
        self, 
        in_channels=3072, 
        mid_channels=[512, 128], 
        out_channels=3,
        upsample_rate=None,
        frame_upsample_rate=[1.0, 1.0],
        joint_upsample_rate=[1.0, 1.0],
        cond_dim=512,  # ⭐ 新增：视觉条件向量的维度
        # 其他你可能需要的参数
        dim=128,
        attention_head_dim=64,
        num_attention_heads=8
        ):
        super(ConditionalDecoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        
        # 使用 ConditionalResBlock 替换 ResBlock
        self.resnet1 = nn.ModuleList([
            ConditionalResBlock(mid_channels[0], mid_channels[0], cond_dim) for _ in range(3)
        ])
        self.upsample1 = Upsample(mid_channels[0], mid_channels[0], frame_upsample_rate=frame_upsample_rate[0], joint_upsample_rate=joint_upsample_rate[0])
        
        # 第一个 ResBlock 改变通道数，也需要是 Conditional
        self.resnet2 = ConditionalResBlock(mid_channels[0], mid_channels[1], cond_dim)
        
        self.resnet3 = nn.ModuleList([
            ConditionalResBlock(mid_channels[1], mid_channels[1], cond_dim) for _ in range(3)
        ])
        self.upsample2 = Upsample(mid_channels[1], mid_channels[1], frame_upsample_rate=frame_upsample_rate[1], joint_upsample_rate=joint_upsample_rate[1])
        
        # 输出前的最后处理
        self.norm_out = nn.GroupNorm(32, mid_channels[-1]) # 使用标准Norm，因为不再有条件输入
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, cond): # ⭐ forward 方法签名改变，增加了 cond
        """
        Args:
            x (torch.Tensor): 来自 VQ 码本的量化向量, shape [B, C_in, H, W]
            cond (torch.Tensor): 视觉条件向量, shape [B, cond_dim]
        """
        x = self.conv_in(x)
        
        for resnet_block in self.resnet1:
            x = resnet_block(x, cond) # ⭐ 将 cond 传入每个残差块
        x = self.upsample1(x)

        x = self.resnet2(x, cond) # ⭐ 将 cond 传入每个残差块
        for resnet_block in self.resnet3:
            x = resnet_block(x, cond) # ⭐ 将 cond 传入每个残差块
        x = self.upsample2(x)

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        return x

