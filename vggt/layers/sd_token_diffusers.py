# SD Token Embedding using Hugging Face Diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
from typing import Dict, Any, Union, Tuple
from torchvision import transforms
import numpy as np

try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available. Install with: pip install diffusers transformers")
    DIFFUSERS_AVAILABLE = False

class SDTokenDiffusers(nn.Module):
    """
    使用Hugging Face Diffusers的SD Token Embedding
    这是一个更简单、更稳定的替代方案
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        unet_layers: Tuple[int, ...] = (2, 5, 8, 11),
        use_text_encoder: bool = False,
        timestep: int = 50,
    ):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.unet_layers = unet_layers
        self.timestep = timestep
        
        # 计算patch数量
        self.num_patches_h = self.img_size[0] // patch_size
        self.num_patches_w = self.img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # 加载SD模型组件
        if DIFFUSERS_AVAILABLE:
            self._load_sd_components(sd_model_id, use_text_encoder)
        else:
            print("Diffusers not available, using dummy features")
            self.unet = None
            self.vae = None
            self.text_encoder = None
            self.tokenizer = None
        
        # 特征维度计算
        # UNet通常每层输出不同的通道数，这里简化为固定维度
        self.sd_feature_dim = 768  # 简化的特征维度
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(self.sd_feature_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 位置编码 - 仿照 vision_transformer.py 的设计
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.interpolate_antialias = False
        self.interpolate_offset = 0.1
        
        # LayerNorm层
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        self._init_weights()
    
    def _load_sd_components(self, model_id, use_text_encoder):
        """加载SD模型组件"""
        try:
            print(f"Loading SD components from {model_id}...")
            
            # 加载UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", torch_dtype=torch.float32
            )
            
            # 加载VAE
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.float32
            )
            
            # 可选：加载文本编码器
            if use_text_encoder:
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_id, subfolder="text_encoder"
                )
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    model_id, subfolder="tokenizer"
                )
            else:
                self.text_encoder = None
                self.tokenizer = None
            
            # 设置为评估模式并冻结参数
            self.unet.eval()
            self.vae.eval()
            for param in self.unet.parameters():
                param.requires_grad = False
            for param in self.vae.parameters():
                param.requires_grad = False
                
            if self.text_encoder is not None:
                self.text_encoder.eval()
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            
            print("SD components loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load SD components: {e}")
            print("Using dummy components")
            self.unet = None
            self.vae = None
            self.text_encoder = None
            self.tokenizer = None
    
    def interpolate_pos_encoding(self, x, w, h):
        """位置编码插值 - 仿照 vision_transformer.py"""
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        
        if npatch == N and w == h:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M, f"Position embedding must be square, got {N} patches"
        
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        
        patch_pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)
    
    def _init_weights(self):
        """初始化权重"""
        trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.feature_projection.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, masks=None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W]
            masks: Optional masks (kept for compatibility)
        
        Returns:
            dict: 与 vision_transformer.py 兼容的字典格式
        """
        B, C, H, W = x.shape
        original_H, original_W = H, W
        
        # 调整输入尺寸
        if (H, W) != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
            H, W = self.img_size
        
        # 提取SD特征
        with torch.no_grad():
            sd_features = self._extract_sd_features_diffusers(x)
        
        # 重塑为patch格式
        patch_tokens = sd_features.permute(0, 2, 3, 1)  # [B, patch_h, patch_w, feature_dim]
        patch_tokens = patch_tokens.reshape(B, -1, sd_features.shape[1])  # [B, num_patches, feature_dim]
        
        # 特征投影
        patch_tokens = self.feature_projection(patch_tokens)
        
        # 添加位置编码
        pos_embed = self.interpolate_pos_encoding(patch_tokens, original_W, original_H)
        patch_tokens = patch_tokens + pos_embed
        
        # 应用LayerNorm
        x_norm = self.norm(patch_tokens)
        
        return {
            "x_norm_clstoken": None,
            "x_norm_regtokens": None,
            "x_norm_patchtokens": x_norm,
            "x_prenorm": patch_tokens,
            "masks": masks,
        }
    
    def _extract_sd_features_diffusers(self, images: torch.Tensor) -> torch.Tensor:
        """使用diffusers提取SD特征"""
        if self.unet is None or self.vae is None:
            # 备用方案：返回随机特征
            print("Warning: SD models not loaded, using random features")
            B = images.shape[0]
            return torch.randn(B, self.sd_feature_dim, self.num_patches_h, self.num_patches_w,
                             device=images.device, dtype=images.dtype)
        
        B = images.shape[0]
        device = images.device
        
        # 将图像编码到潜在空间
        # 注意：这里简化了处理，实际应用中可能需要更复杂的预处理
        images_normalized = images * 2.0 - 1.0  # 转换到[-1, 1]范围
        
        # 使用VAE编码器
        latents = self.vae.encode(images_normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # 添加噪声（可选）
        noise = torch.randn_like(latents)
        timesteps = torch.full((B,), self.timestep, device=device, dtype=torch.long)
        
        # 使用调度器添加噪声（这里简化处理）
        alpha = 0.5  # 简化的噪声系数
        noisy_latents = alpha * latents + (1 - alpha) * noise
        
        # 获取无条件文本嵌入
        if self.text_encoder is not None:
            uncond_tokens = self.tokenizer(
                [""] * B, padding="max_length", max_length=77, 
                truncation=True, return_tensors="pt"
            )
            text_embeddings = self.text_encoder(uncond_tokens.input_ids.to(device))[0]
        else:
            # 使用零嵌入
            text_embeddings = torch.zeros(B, 77, 768, device=device)
        
        # UNet前向传播并提取中间特征
        features = self._extract_unet_features(noisy_latents, timesteps, text_embeddings)
        
        return features
    
    def _extract_unet_features(self, latents, timesteps, encoder_hidden_states):
        """从UNet提取中间特征"""
        # 这是一个简化的实现，实际中可能需要hook机制来提取中间特征
        # 这里我们使用UNet的输出作为特征
        
        try:
            # 获取UNet输出
            noise_pred = self.unet(latents, timesteps, encoder_hidden_states).sample
            
            # 简化处理：直接使用输出特征
            # 在实际应用中，您可能想要提取UNet的中间层特征
            features = noise_pred
            
            # 调整特征维度和分辨率以匹配patch要求
            target_size = (self.num_patches_h, self.num_patches_w)
            if features.shape[-2:] != target_size:
                features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
            
            # 调整通道维度
            if features.shape[1] != self.sd_feature_dim:
                # 使用卷积调整通道数
                if not hasattr(self, 'channel_adapter'):
                    self.channel_adapter = nn.Conv2d(
                        features.shape[1], self.sd_feature_dim, 1
                    ).to(features.device)
                features = self.channel_adapter(features)
            
            return features
            
        except Exception as e:
            print(f"Error in UNet feature extraction: {e}")
            # 备用方案
            B = latents.shape[0]
            return torch.randn(B, self.sd_feature_dim, self.num_patches_h, self.num_patches_w,
                             device=latents.device, dtype=latents.dtype)


# 简化的安装指南
def install_requirements():
    """安装所需依赖的指南"""
    print("""
    To use SDTokenDiffusers, install the required packages:
    
    pip install diffusers transformers accelerate
    
    Optional for better performance:
    pip install xformers
    """)

if __name__ == "__main__":
    if not DIFFUSERS_AVAILABLE:
        install_requirements()
    else:
        # 测试代码
        model = SDTokenDiffusers(img_size=518, patch_size=14, embed_dim=1024)
        test_input = torch.randn(1, 3, 518, 518)
        output = model(test_input)
        print(f"Output shape: {output['x_norm_patchtokens'].shape}")
