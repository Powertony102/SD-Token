# SD Token 替代方案指南

由于ODISE环境配置复杂，我们提供了两个更简单的Stable Diffusion token替代方案。

## 🎯 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **SDTokenDiffusers** | 使用官方SD模型，特征质量高 | 需要安装diffusers库 | 追求最佳效果 |
| **SDTokenSimple** | 无外部依赖，安装简单 | 特征质量略低于真实SD | 快速原型和测试 |

## 🚀 使用方法

### 方案1: SDTokenDiffusers（推荐）

#### 安装依赖
```bash
pip install diffusers transformers accelerate

# 可选：安装xformers以提高性能
pip install xformers
```

#### 使用方法
```python
from vggt.models.vggt import VGGT

# 使用Diffusers版本的SD Token
model = VGGT(
    img_size=518,
    patch_size=14, 
    embed_dim=1024,
    patch_embed="sd_diffusers"  # 使用diffusers版本
)

# 前向传播
images = torch.randn(2, 4, 3, 518, 518)  # [B, S, C, H, W]
output = model(images)
```

### 方案2: SDTokenSimple（无依赖）

#### 使用方法
```python
from vggt.models.vggt import VGGT

# 使用简化版本的SD Token
model = VGGT(
    img_size=518,
    patch_size=14,
    embed_dim=1024, 
    patch_embed="sd_simple"  # 或者直接用 "sd"
)

# 前向传播
images = torch.randn(2, 4, 3, 518, 518)
output = model(images)
```

## 🔧 技术细节

### SDTokenDiffusers 特点
- ✅ 使用真实的SD UNet和VAE
- ✅ 支持多种SD模型版本
- ✅ 特征质量接近真实SD
- ✅ 完整的位置编码插值
- ❌ 需要额外的依赖包

### SDTokenSimple 特点  
- ✅ 零外部依赖（除了PyTorch）
- ✅ 使用ResNet50模拟SD特征
- ✅ 完整的位置编码系统
- ✅ 与VGGT完全兼容
- ❌ 特征质量不如真实SD

## 📊 性能对比

| 指标 | SDTokenDiffusers | SDTokenSimple | DINOv2 |
|------|------------------|---------------|---------|
| **加载时间** | ~10s | ~2s | ~5s |
| **内存使用** | ~4GB | ~1GB | ~2GB |
| **推理速度** | 中等 | 快 | 快 |
| **特征质量** | 高 | 中等 | 高 |

## 🛠 故障排除

### 常见问题

1. **ImportError: No module named 'diffusers'**
   ```bash
   pip install diffusers transformers
   ```

2. **CUDA out of memory**
   - 减小batch size
   - 使用`torch_dtype=torch.float16`
   - 使用CPU模式

3. **模型下载慢**
   - 使用国内镜像：`export HF_ENDPOINT=https://hf-mirror.com`
   - 或手动下载模型到本地

### 自定义配置

#### 使用不同的SD模型
```python
# 在sd_token_diffusers.py中修改
sd_model_id = "stabilityai/stable-diffusion-2-1"  # 使用SD 2.1
# 或者
sd_model_id = "runwayml/stable-diffusion-v1-5"   # 使用SD 1.5
```

#### 调整特征提取层
```python
# 修改UNet层选择
unet_layers = (2, 5, 8, 11)  # 默认层
unet_layers = (1, 3, 6, 9)   # 其他层组合
```

## 🎨 使用建议

### 推荐配置

**追求效果（有GPU资源）**:
```python
model = VGGT(patch_embed="sd_diffusers")
```

**快速开发和测试**:
```python  
model = VGGT(patch_embed="sd_simple")
```

**混合使用**:
```python
# 开发阶段用simple，部署时用diffusers
patch_embed = "sd_simple" if args.debug else "sd_diffusers"
model = VGGT(patch_embed=patch_embed)
```

## 📈 未来改进

1. **特征缓存**: 缓存提取的SD特征以提高推理速度
2. **量化支持**: 支持INT8量化减少内存使用
3. **更多SD版本**: 支持SDXL、SD 2.1等新版本
4. **特征融合**: 结合DINOv2和SD特征的混合模式

## 🤝 贡献

欢迎提交PR改进这些替代方案！特别是：
- 性能优化
- 新的SD模型支持  
- 更好的特征提取策略
- 文档改进
