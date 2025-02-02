import argparse  # 导入argparse模块，用于命令行参数解析
import os  # 导入os模块，用于操作系统相关功能
import torch  # 导入PyTorch库
from exp.exp_main import Exp_Main  # 从exp_main模块中导入Exp_Main类
import random  # 导入random模块，用于生成随机数
import numpy as np  # 导入NumPy库，用于科学计算
import sys  # 导入sys模块，用于系统相关操作
from tqdm import tqdm  # 导入tqdm库
os.chdir(sys.path[0])  # 将当前工作目录更改为脚本所在目录，以确保相对路径正确

# 创建ArgumentParser对象，用于处理命令行参数
parser = argparse.ArgumentParser(description='ConvTimeNet for Time Series Forecasting')

# 设置随机种子参数，确保实验的可重复性
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# 基本配置参数
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')  # 是否处于训练模式
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')  # 模型标识符
parser.add_argument('--model', type=str, required=False, default='ConvTimeNet',
                    help='model name, options: [Autoformer, Informer, Transformer]')  # 模型名称

# 数据加载器参数
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')  # 数据集类型
parser.add_argument('--root_path', type=str, default='../dataset/', help='root path of the data file')  # 数据文件根路径
# parser.add_argument('--root_path', type=str, default='E:/浏览器下载地址/宁海疾控食源性疾病/TimeNet/dataset/', help='root path of the data file')  # 数据文件根路径
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  # 数据文件名
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  # 数据文件名
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')  # 预测任务类型
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')  # 目标特征
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')  # 时间特征编码的频率
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')  # 模型检查点位置

# 预测任务参数
parser.add_argument('--seq_len', type=int, default=2, help='input sequence length')  # 输入序列长度
parser.add_argument('--label_len', type=int, default=1, help='start token length')  # 标签长度
parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')  # 预测序列长度

# ConvTimeNet相关参数
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')  # 头部dropout
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')  # 补丁填充方式
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')  # 是否使用RevIN
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')  # RevIN的仿射变换
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')  # 减去最后一个值或均值

parser.add_argument('--dw_ks', type=str, default='11,15,21,29,39,51', help="kernel size of the deep-wise. default:9")  # 深度卷积核大小
parser.add_argument('--re_param', type=int, default=1, help='Reparam the DeepWise Conv when train')  # 训练时重新参数化深度卷积
parser.add_argument('--enable_res_param', type=int, default=1, help='Learnable residual')  # 可学习的残差
parser.add_argument('--re_param_kernel', type=int, default=3)  # 重新参数化的卷积核大小

# 补丁相关参数
parser.add_argument('--patch_ks', type=int, default=3, help="kernel size of the patch window. default:32")  # 补丁窗口的卷积核大小
parser.add_argument('--patch_sd', type=float, default=0.5, \
                    help="stride of the patch window. default: 0.5. if < 1, then sd = patch_sd * patch_ks")  # 补丁窗口的步幅

# 其他参数
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')  # 编码器输入大小
parser.add_argument('--c_out', type=int, default=7, help='output size')  # 输出大小
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')  # 模型维度
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')  # 编码器层数
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')  # 全连接层的维度

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')  # dropout率
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')  # 时间特征编码方式
parser.add_argument('--activation', type=str, default='gelu', help='activation')  # 激活函数
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')  # 是否预测未见的未来数据

# 优化参数
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')  # 数据加载器的工作线程数
parser.add_argument('--itr', type=int, default=2, help='experiments times')  # 实验次数
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')  # 训练轮数
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')  # 训练输入数据的批次大小
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # 提前停止的耐心
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')  # 优化器学习率
parser.add_argument('--des', type=str, default='test', help='exp description')  # 实验描述
parser.add_argument('--loss', type=str, default='mse', help='loss function')  # 损失函数
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)  # 是否使用自动混合精度训练
# 在数据加载器参数部分添加 scale 参数
parser.add_argument('--scale', type=bool, default=True, help='whether to scale the data')  # 数据是否标准化

# GPU相关参数
parser.add_argument('--CTX', type=str, default='0', required=False, help='visuable device ids')  # 可见设备ID
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')  # 是否使用GPU
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')  # 是否使用GPU
parser.add_argument('--gpu', type=int, default=0, help='gpu')  # GPU编号
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)  # 是否使用多GPU
# parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')  # 多GPU的设备ID
parser.add_argument('--devices', type=str, default='cpu', help='device ids of multile gpus')  # 多GPU的设备ID
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')  # 是否测试FLOP

if __name__ == '__main__':
    # 解析命令行参数
    args = parser.parse_args()

    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CTX

    # 设置随机种子，确保实验的可重复性
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 检查是否使用GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 如果使用多GPU
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')  # 去除设备ID中的空格
        device_ids = args.devices.split(',')  # 分割设备ID
        args.device_ids = [int(id_) for id_ in device_ids]  # 转换为整数列表
        args.gpu = args.device_ids[0]  # 设置主GPU

    # 解析深度卷积核大小
    args.dw_ks = [int(ks) for ks in args.dw_ks.split(',')]
    # 计算补丁步幅
    args.patch_sd = max(1, int(args.patch_ks * args.patch_sd)) if args.patch_sd <= 1 else int(args.patch_sd)
    # 确保编码器层数与深度卷积核列表长度匹配
    assert args.e_layers == len(args.dw_ks), "e_layers should match the dw kernel list!"

    # 打印实验参数
    print('Args in experiment:')
    print(args)

    # 实验主类
    Exp = Exp_Main
    mses = []  # 存储均方误差
    maes = []  # 存储平均绝对误差

    # 如果是训练模式
    if args.is_training:
        for ii in range(args.itr):
            # 设置实验记录
            setting = '{}_dm{}_df{}_el{}_dk{}_pk{}_ps{}_erp{}_rp{}_{}_{}'.format(
                args.model_id,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.dw_ks, 
                args.patch_ks,
                args.patch_sd,
                args.enable_res_param,
                args.re_param,
                args.des,ii)

            exp = Exp(args)  # 设置实验
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            
            # 使用tqdm显示训练进度
            for epoch in tqdm(range(args.train_epochs), desc="Training Progress"):
                exp.train(setting)  # 开始训练

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            # 使用tqdm显示测试进度
            for _ in tqdm(range(1), desc="Testing Progress"):
                mse, mae = exp.test(setting)  # 测试模型
            
            mses.append(mse)  # 记录均方误差
            maes.append(mae)  # 记录平均绝对误差

            torch.cuda.empty_cache()  # 清空CUDA缓存

        # 打印平均误差
        print('average mse:{0:.3f}±{1:.3f}, mae:{2:.3f}±{3:.3f}'.format(np.mean(
        mses), np.std(mses), np.mean(maes), np.std(maes))) 
    else:
        ii = 0
        setting = '{}_dm{}_df{}_el{}_dk{}_pk{}_ps{}_erp{}_rp{}_{}_{}'.format(
            args.model_id,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.dw_ks, 
            args.patch_ks,
            args.patch_sd,
            args.enable_res_param,
            args.re_param,
            args.des,ii)

        exp = Exp(args)  # 设置实验
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        # 使用tqdm显示测试进度
        for _ in tqdm(range(1), desc="Testing Progress"):
            exp.test(setting, test=1)  # 测试模型
        
        torch.cuda.empty_cache()  # 清空CUDA缓存
