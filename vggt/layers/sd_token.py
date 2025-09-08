# SD Token Embedding for VGGT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
from typing import Dict, Any, Union, Tuple
from torchvision import transforms
import sys
import os

# 添加SD-DINO路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from extractor_sd import load_model, process_features_and_mask

class SDPatchEmbed(nn.Module):
    """
    Stable Diffusion Patch Embedding for VGGT
    
    将SD特征转换为与DINOv2兼容的patch embedding格式
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        sd_version: str = "v1-5",
        sd_timestep: int = 100,
        sd_block_indices: Tuple[int, ...] = (2, 5, 8, 11),
        use_pca: bool = True,
        pca_dims: Tuple[int, ...] = (256, 256, 256),
        mask_enabled: bool = False,
    ):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_pca = use_pca
        self.mask_enabled = mask_enabled
        
        # 计算patch数量
        self.num_patches_h = self.img_size[0] // patch_size
        self.num_patches_w = self.img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # 加载SD模型
        self.sd_model, self.sd_aug = load_model(
            diffusion_ver=sd_version,
            image_size=max(self.img_size),
            num_timesteps=sd_timestep,
            block_indices=sd_block_indices
        )
        
        # 计算SD特征维度
        if use_pca:
            sd_feature_dim = sum(pca_dims)  # 默认256+256+256=768
        else:
            # 根据block_indices计算原始特征维度
            sd_feature_dim = self._calculate_sd_feature_dim(sd_block_indices)
        
        # 特征投影层：将SD特征投影到目标维度
        self.feature_projection = nn.Sequential(
            nn.Linear(sd_feature_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 位置编码 - 仿照 vision_transformer.py 的设计
        # 注意：这里不包含 cls_token，只有 patch tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 位置编码插值相关参数 (仿照 DinoVisionTransformer)
        self.interpolate_antialias = False
        self.interpolate_offset = 0.1
        
        # LayerNorm层 (仿照 vision_transformer.py)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        self._init_weights()
    
    def _calculate_sd_feature_dim(self, block_indices):
        """计算SD特征的原始维度"""
        # 这里需要根据实际的SD特征维度来计算
        # 暂时使用经验值
        return 768  # 可能需要根据实际情况调整
    
    def interpolate_pos_encoding(self, x, w, h):
        """
        位置编码插值方法 - 仿照 vision_transformer.py 中的实现
        当输入图像尺寸与预训练尺寸不同时，对位置编码进行插值
        """
        import math
        
        previous_dtype = x.dtype
        npatch = x.shape[1]  # 当前patch数量
        N = self.pos_embed.shape[1]  # 预训练时的patch数量
        
        # 如果patch数量相同且图像是正方形，直接返回
        if npatch == N and w == h:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        
        # 计算当前的patch网格尺寸
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        
        # 计算原始的patch网格尺寸 (假设是正方形)
        M = int(math.sqrt(N))
        assert N == M * M, f"Position embedding must be square, got {N} patches"
        
        kwargs = {}
        if self.interpolate_offset:
            # 添加小的偏移以避免插值时的浮点误差
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # 直接指定输出尺寸
            kwargs["size"] = (w0, h0)
        
        # 对位置编码进行双三次插值
        patch_pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        
        assert (w0, h0) == patch_pos_embed.shape[-2:], f"Interpolation failed: expected {(w0, h0)}, got {patch_pos_embed.shape[-2:]}"
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return patch_pos_embed.to(previous_dtype)
    
    def _init_weights(self):
        """初始化权重 - 仿照 vision_transformer.py"""
        # 使用截断正态分布初始化位置编码
        trunc_normal_(self.pos_embed, std=0.02)
        
        # 初始化特征投影层
        for m in self.feature_projection.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, masks=None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W]
            masks: Optional masks for patches (not used in SD, kept for compatibility)
        
        Returns:
            dict: 与 vision_transformer.py 兼容的字典格式
                - "x_norm_clstoken": None (SD没有cls token)
                - "x_norm_regtokens": None (SD没有register tokens)  
                - "x_norm_patchtokens": [B, num_patches, embed_dim]
                - "x_prenorm": [B, num_patches, embed_dim] (与normalized相同)
                - "masks": masks (传入的masks参数)
        """
        B, C, H, W = x.shape
        original_H, original_W = H, W
        
        # 确保输入尺寸正确
        if (H, W) != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
            H, W = self.img_size
        
        # 提取SD特征
        with torch.no_grad():
            sd_features = self._extract_sd_features(x)
        
        # 重塑为patch格式
        # sd_features: [B, feature_dim, patch_h, patch_w]
        patch_tokens = sd_features.permute(0, 2, 3, 1)  # [B, patch_h, patch_w, feature_dim]
        patch_tokens = patch_tokens.reshape(B, -1, sd_features.shape[1])  # [B, num_patches, feature_dim]
        
        # 特征投影
        patch_tokens = self.feature_projection(patch_tokens)
        
        # 添加位置编码 - 使用插值方法处理不同尺寸
        pos_embed = self.interpolate_pos_encoding(patch_tokens, original_W, original_H)
        patch_tokens = patch_tokens + pos_embed
        
        # 应用LayerNorm (仿照 vision_transformer.py 的 norm 层)
        x_norm = self.norm(patch_tokens)
        
        # 返回与 vision_transformer.py 兼容的字典格式
        return {
            "x_norm_clstoken": None,  # SD没有cls token
            "x_norm_regtokens": None,  # SD没有register tokens
            "x_norm_patchtokens": x_norm,  # 标准化后的patch tokens
            "x_prenorm": patch_tokens,     # 标准化前的patch tokens
            "masks": masks,               # 传入的masks (可能为None)
        }
    
    def _extract_sd_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取SD特征
        
        Args:
            images: [B, 3, H, W]
        
        Returns:
            features: [B, feature_dim, patch_h, patch_w]
        """
        batch_features = []
        
        for i in range(images.shape[0]):
            # 转换为PIL格式
            img_pil = transforms.ToPILImage()(images[i].cpu())
            
            # 提取SD特征
            features = process_features_and_mask(
                self.sd_model, 
                self.sd_aug, 
                img_pil,
                mask=self.mask_enabled,
                pca=self.use_pca,
                raw=not self.use_pca
            )
            
            batch_features.append(features)
        
        # 堆叠批次
        batch_features = torch.stack(batch_features, dim=0)
        
        # 调整到目标patch尺寸
        target_size = (self.num_patches_h, self.num_patches_w)
        if batch_features.shape[-2:] != target_size:
            batch_features = F.interpolate(
                batch_features, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return batch_features