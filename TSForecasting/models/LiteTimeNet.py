import torch
import torch.nn as nn

class LiteTimeNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 基本参数
        c_in = configs.enc_in
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        
        # 轻量化参数设置
        d_model = 32       # 减少模型维度
        ff_dim = 64        # 缩小前馈网络维度
        n_layers = 3       # 减少层数
        patch_len = 3      # 更小的补丁尺寸
        stride = 1         # 固定步长
        
        # 精简的归一化层
        self.norm = nn.BatchNorm1d(c_in)
        
        # 高效补丁处理
        self.patch_embed = nn.Sequential(
            nn.Conv1d(c_in, d_model, patch_len, stride, padding='same'),
            nn.GELU(),
        )
        
        # 轻量卷积模块堆叠
        self.blocks = nn.Sequential(*[
            ConvBlock(d_model, ff_dim, kernel_size=5)  # 统一卷积核尺寸
            for _ in range(n_layers)
        ])
        
        # 简化预测头
        self.head = nn.Conv1d(d_model, c_in, pred_len, padding='same')
    
    def forward(self, x):
        # x shape: [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        
        # 归一化
        x = self.norm(x)
        
        # 特征嵌入
        x = self.patch_embed(x)  # [B, D, L]
        
        # 卷积处理
        x = self.blocks(x)       # [B, D, L]
        
        # 最终预测
        x = self.head(x)         # [B, C, L]
        return x.permute(0, 2, 1)  # [B, L, C]

class ConvBlock(nn.Module):
    def __init__(self, d_model, ff_dim, kernel_size=5):
        super().__init__()
        # 深度可分离卷积
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding='same', groups=d_model
        )
        self.act = nn.GELU()
        
        # 轻量前馈网络
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, ff_dim, 1),
            nn.GELU(),
            nn.Conv1d(ff_dim, d_model, 1),
        )
        
        # 精简归一化
        self.norm = nn.BatchNorm1d(d_model)
    
    def forward(self, x):
        # 残差连接1
        res = x
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.norm(x + res)
        
        # 残差连接2
        res = x
        x = self.ff(x)
        return x + res



# 测试
import torch
from torch import nn

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
    configs = Configs(enc_in=10, seq_len=50, pred_len=10)  # 示例配置
    model = LiteTimeNet(configs).to(device)

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