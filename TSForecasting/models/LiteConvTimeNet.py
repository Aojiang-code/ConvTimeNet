__all__ = ['LiteConvTimeNet']

import torch
from torch import nn

class LiteConvTimeNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 基础配置
        c_in = configs.enc_in
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        
        # 轻量化参数设置
        d_model = 64  # 减少特征维度
        n_layers = 3  # 减少层数
        patch_len = 8  # 增大补丁长度
        stride = 4    # 增大步长
        
        # 核心模块
        self.revin = RevIN(c_in)  # 简化RevIN去除仿射变换
        self.patch_embed = nn.Conv1d(c_in, d_model, patch_len, stride)  # 使用普通卷积替代可变形采样
        self.encoder = LiteConvEncoder(d_model, n_layers, kernel_size=15)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * ((seq_len - patch_len) // stride + 1), pred_len * c_in),
            nn.Unflatten(1, (pred_len, c_in))  # 将输出张量的维度从 [B, pred_len * c_in] 转换为 [B, pred_len, c_in]
        )

        # 计算特征提取后的序列长度
        output_len = (seq_len - patch_len) // stride + 1
        print(f"Output length after patch embedding: {output_len}")

    def forward(self, x):
        # 打印输入张量的形状
        print(f"Input tensor shape: {x.shape}")
        
        # 确保输入张量的维度是 [B, L, C]
        assert x.dim() == 3, f"Input tensor should have 3 dimensions, but got {x.dim()} dimensions."
        
        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        
        # 归一化
        x = self.revin(x, 'norm')
        
        # 特征提取
        x = self.patch_embed(x)  # [B, D, N]
        x = self.encoder(x)
        
        # 确保特征张量的维度是 [B, D, N]
        assert x.dim() == 3, f"Feature tensor should have 3 dimensions, but got {x.dim()} dimensions."
        print(f"Feature tensor shape: {x.shape}")

        # 预测头
        x = self.head(x.permute(0, 2, 1))  # [B, L, C]
        
        # 确保预测结果的维度是 [B, L, C]
        assert x.dim() == 3, f"Output tensor should have 3 dimensions, but got {x.dim()} dimensions."
        
        # 反归一化
        x = self.revin(x.permute(0, 2, 1), 'denorm')  # 确保维度一致
        
        # 打印最终输出张量的形状
        print(f"Output tensor shape: {x.shape}")
        
        return x.permute(0, 2, 1)  # [B, L, C]

class LiteConvEncoder(nn.Module):
    def __init__(self, d_model, n_layers, kernel_size):
        super().__init__()
        self.layers = nn.ModuleList([
            LiteConvBlock(d_model, kernel_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LiteConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        # 深度可分离卷积
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding='same', groups=d_model
        )
        self.act = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, 1),
            nn.GELU(),
            nn.Conv1d(d_model*2, d_model, 1)
        )
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        res = x
        # 深度卷积分支
        x = self.dw_conv(x)
        x = self.act(x)
        # 前馈分支
        x = self.ffn(x)
        # 残差连接
        return self.norm(res + x)

class RevIN(nn.Module):
    """简化版RevIN去除了仿射变换"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.stdev = None

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = x - self.mean
            x = x / (self.stdev + self.eps)
        elif mode == 'denorm':
            x = x * self.stdev
            x = x + self.mean
        return x

    def _get_statistics(self, x):
        # 确保 mean 和 stdev 的维度是 [batch_size, num_features, 1]
        self.mean = x.mean(dim=2, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=2, keepdim=True, unbiased=False) + self.eps).detach()

# 模型配置类
class Configs:
    def __init__(self, enc_in, seq_len, pred_len):
        self.enc_in = enc_in  # 输入特征维度
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 预测序列长度

# 测试函数
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型配置
    configs = Configs(enc_in=10, seq_len=100, pred_len=10)  # 示例配置
    model = LiteConvTimeNet(configs).to(device)

    # 创建一个随机输入张量
    batch_size = 4  # 示例批量大小
    input_data = torch.randn(batch_size, configs.seq_len, configs.enc_in).to(device)

    # 模型前向传播
    try:
        output = model(input_data)
        print(f"Model output shape: {output.shape}")
        print("Model runs successfully!")
    except Exception as e:
        print(f"Model failed to run: {e}")