import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')  # 忽略所有警告信息

# 定义一个用于小时级别数据集的类
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # 初始化信息
        if size == None:
            self.seq_len = 24 * 4 * 4  # 默认序列长度为16天
            self.label_len = 24 * 4  # 默认标签长度为1天
            self.pred_len = 24 * 4  # 默认预测长度为1天
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # 初始化
        assert flag in ['train', 'test', 'val']  # 确保标志在指定范围内
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 设置数据集类型

        self.features = features  # 特征类型
        self.target = target  # 目标变量
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc  # 时间编码类型
        self.freq = freq  # 数据频率

        self.root_path = root_path  # 数据集根路径
        self.data_path = data_path  # 数据文件路径
        self.__read_data__()  # 读取数据

    def __read_data__(self):
        self.scaler = StandardScaler()  # 初始化标准化缩放器
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  # 读取CSV数据

        # 定义数据集的边界
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]  # 根据数据集类型选择边界
        border2 = border2s[self.set_type]

        # 根据特征类型选择数据列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 多变量特征
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]  # 单变量特征

        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # 仅对训练集进行标准化
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # 转换为日期时间格式
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values  # 删除日期列
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # 转置时间特征

        self.data_x = data[border1:border2]  # 输入数据
        self.data_y = data[border1:border2]  # 输出数据
        self.data_stamp = data_stamp  # 时间标记

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # 输入序列
        seq_y = self.data_y[r_begin:r_end]  # 输出序列
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 输入时间标记
        seq_y_mark = self.data_stamp[r_begin:r_end]  # 输出时间标记

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1  # 数据集长度

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)  # 逆标准化

# 定义一个用于分钟级别数据集的类
class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # 初始化信息
        if size == None:
            self.seq_len = 24 * 4 * 4  # 默认序列长度为16天
            self.label_len = 24 * 4  # 默认标签长度为1天
            self.pred_len = 24 * 4  # 默认预测长度为1天
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # 初始化
        assert flag in ['train', 'test', 'val']  # 确保标志在指定范围内
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 设置数据集类型

        self.features = features  # 特征类型
        self.target = target  # 目标变量
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc  # 时间编码类型
        self.freq = freq  # 数据频率

        self.root_path = root_path  # 数据集根路径
        self.data_path = data_path  # 数据文件路径
        self.__read_data__()  # 读取数据

    def __read_data__(self):
        self.scaler = StandardScaler()  # 初始化标准化缩放器
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  # 读取CSV数据

        # 定义数据集的边界
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]  # 根据数据集类型选择边界
        border2 = border2s[self.set_type]

        # 根据特征类型选择数据列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 多变量特征
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]  # 单变量特征

        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # 仅对训练集进行标准化
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # 转换为日期时间格式
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)  # 将分钟转换为15分钟间隔
            data_stamp = df_stamp.drop(['date'], 1).values  # 删除日期列
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # 转置时间特征

        self.data_x = data[border1:border2]  # 输入数据
        self.data_y = data[border1:border2]  # 输出数据
        self.data_stamp = data_stamp  # 时间标记

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # 输入序列
        seq_y = self.data_y[r_begin:r_end]  # 输出序列
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 输入时间标记
        seq_y_mark = self.data_stamp[r_begin:r_end]  # 输出时间标记

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1  # 数据集长度

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)  # 逆标准化

# 定义一个用于自定义数据集的类
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # 初始化信息
        if size == None:
            self.seq_len = 24 * 4 * 4  # 默认序列长度为16天
            self.label_len = 24 * 4  # 默认标签长度为1天
            self.pred_len = 24 * 4  # 默认预测长度为1天
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # 初始化
        assert flag in ['train', 'test', 'val']  # 确保标志在指定范围内
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 设置数据集类型
        self.train_only = train_only  # 是否仅使用训练集

        self.features = features  # 特征类型
        self.target = target  # 目标变量
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc  # 时间编码类型
        self.freq = freq  # 数据频率

        self.root_path = root_path  # 数据集根路径
        self.data_path = data_path  # 数据文件路径
        self.__read_data__()  # 读取数据

    def __read_data__(self):
        self.scaler = StandardScaler()  # 初始化标准化缩放器
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  # 读取CSV数据

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)  # 移除目标列
        cols.remove('date')  # 移除日期列

        # 计算训练、测试和验证集的大小
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))  # 训练集大小
        num_test = int(len(df_raw) * 0.2)  # 测试集大小
        num_vali = len(df_raw) - num_train - num_test  # 验证集大小
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]  # 根据数据集类型选择边界
        border2 = border2s[self.set_type]

        # 根据特征类型选择数据列
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.num_features = len(cols_data)  # 特征数量
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]
            self.num_features = 1  # 单变量特征数量

        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # 仅对训练集进行标准化
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # 转换为日期时间格式
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values  # 删除日期列
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # 转置时间特征

        self.data_x = data[border1:border2]  # 输入数据
        self.data_y = data[border1:border2]  # 输出数据
        self.data_stamp = data_stamp  # 时间标记

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # 输入序列
        seq_y = self.data_y[r_begin:r_end]  # 输出序列
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 输入时间标记
        seq_y_mark = self.data_stamp[r_begin:r_end]  # 输出时间标记

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1  # 数据集长度

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)  # 逆标准化

# 定义一个用于预测数据集的类
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # 初始化信息
        if size == None:
            self.seq_len = 24 * 4 * 4  # 默认序列长度
            self.label_len = 24 * 4  # 默认标签长度
            self.pred_len = 24 * 4  # 默认预测长度
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # 初始化
        assert flag in ['pred']  # 确保标志为预测

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()  # 读取数据

    def __read_data__(self):
        self.scaler = StandardScaler()  # 标准化缩放器
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  # 读取CSV数据
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # 根据特征类型选择数据列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 数据标准化
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 处理时间戳
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)