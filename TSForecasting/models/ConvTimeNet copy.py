__all__ = ['ConvTimeNet']

# 导入必要的库
import torch
from torch import nn
from layers.ConvTimeNet_backbone import ConvTimeNet_backbone

# ConvTimeNet: depatch + batch norm + gelu + Conv + 2-layer-ffn(PointWise Conv + PointWise Conv)
class Model(nn.Module):
    def __init__(self, configs, norm:str='batch', act:str="gelu", head_type='flatten'):
        super().__init__()
        
        # 加载参数
        c_in = configs.enc_in  # 输入通道数
        context_window = configs.seq_len  # 上下文窗口大小
        target_window = configs.pred_len  # 目标窗口大小
        
        n_layers = configs.e_layers  # 网络层数
        d_model = configs.d_model  # 模型维度
        d_ff = configs.d_ff  # 前馈网络维度
        dropout = configs.dropout  # dropout 概率
        head_dropout = configs.head_dropout  # 头部 dropout 概率
    
        patch_len = configs.patch_ks  # patch 大小
        stride = configs.patch_sd  # 步幅
        padding_patch = configs.padding_patch  # patch 填充方式
        
        revin = configs.revin  # 是否使用逆归一化
        affine = configs.affine  # 是否使用仿射变换
        subtract_last = configs.subtract_last  # 是否减去最后一个值
        
        seq_len = configs.seq_len  # 序列长度
        dw_ks = configs.dw_ks  # 深度可分离卷积核大小

        re_param = configs.re_param  # 重新参数化
        re_param_kernel = configs.re_param_kernel  # 重新参数化卷积核大小
        enable_res_param = configs.enable_res_param  # 是否启用残差参数化
    
        # 初始化模型
        self.model = ConvTimeNet_backbone(
            c_in=c_in, 
            seq_len=seq_len, 
            context_window=context_window,
            target_window=target_window, 
            patch_len=patch_len, 
            stride=stride, 
            n_layers=n_layers, 
            d_model=d_model, 
            d_ff=d_ff, 
            dw_ks=dw_ks, 
            norm=norm, 
            dropout=dropout, 
            act=act,
            head_dropout=head_dropout, 
            padding_patch=padding_patch, 
            head_type=head_type, 
            revin=revin, 
            affine=affine, 
            deformable=True, 
            subtract_last=subtract_last, 
            enable_res_param=enable_res_param, 
            re_param=re_param, 
            re_param_kernel=re_param_kernel
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整张量维度顺序
        x = self.model(x)  # 通过模型进行前向传播
        x = x.permute(0, 2, 1)  # 恢复张量维度顺序

        return x  # 返回输出