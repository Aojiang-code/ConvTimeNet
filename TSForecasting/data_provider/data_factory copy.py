from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

# 数据集字典，映射数据集名称到相应的数据集类
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

# 数据提供者函数，根据参数和标志返回数据集和数据加载器
def data_provider(args, flag):
    # 根据参数选择数据集类
    Data = data_dict[args.data]
    # 根据嵌入类型设置时间编码
    timeenc = 0 if args.embed != 'timeF' else 1

    # 根据标志设置不同的参数
    if flag == 'test':
        shuffle_flag = False  # 测试集不打乱
        drop_last = True  # 丢弃最后一个不完整的批次
        batch_size = args.batch_size  # 批次大小
        freq = args.freq  # 数据频率
    elif flag == 'pred':
        shuffle_flag = False  # 预测集不打乱
        drop_last = False  # 不丢弃最后一个批次
        batch_size = 1  # 预测时批次大小为1
        freq = args.freq  # 数据频率
        Data = Dataset_Pred  # 使用预测数据集类
    else:
        shuffle_flag = True  # 训练集打乱
        drop_last = True  # 丢弃最后一个不完整的批次
        batch_size = args.batch_size  # 批次大小
        freq = args.freq  # 数据频率

    # 创建数据集实例
    data_set = Data(
        root_path=args.root_path,  # 数据集根路径
        data_path=args.data_path,  # 数据文件路径
        flag=flag,  # 数据集标志
        size=[args.seq_len, args.label_len, args.pred_len],  # 序列长度、标签长度、预测长度
        features=args.features,  # 特征
        target=args.target,  # 目标变量
        timeenc=timeenc,  # 时间编码
        freq=freq  # 数据频率
    )
    
    print(flag, len(data_set))  # 打印数据集标志和长度
    if len(data_set) < batch_size: 
        batch_size = len(data_set)  # 如果数据集小于批次大小，调整批次大小
    # 创建数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,  # 批次大小
        shuffle=shuffle_flag,  # 是否打乱数据
        num_workers=args.num_workers,  # 工作线程数
        drop_last=drop_last)  # 是否丢弃最后一个不完整的批次
    return data_set, data_loader  # 返回数据集和数据加载器
