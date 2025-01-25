import pandas as pd

# 读取2016_2024.csv文件
file_2016_2024 = r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\2016_2024.csv"
df_2016_2024 = pd.read_csv(file_2016_2024)

# 确保“就诊时间”列是日期时间格式
df_2016_2024['就诊时间'] = pd.to_datetime(df_2016_2024['就诊时间'], format='%Y-%m-%d %H')

# 计算每天的就诊人数
daily_visits = df_2016_2024.groupby(df_2016_2024['就诊时间'].dt.date).size().reset_index(name='就诊人数')

# 读取2017-2023.csv文件
file_2017_2023 = r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2017-2023.csv"
df_2017_2023 = pd.read_csv(file_2017_2023)

# 确保'datetime'列存在并是日期格式
if 'datetime' in df_2017_2023.columns:
    df_2017_2023['datetime'] = pd.to_datetime(df_2017_2023['datetime'])
else:
    raise ValueError("2017-2023.csv文件中没有找到'datetime'列")

# 合并数据
merged_df = pd.merge(df_2017_2023, daily_visits, left_on=df_2017_2023['datetime'].dt.date, right_on='就诊时间', how='left')

# 填充没有就诊记录的日期为0
merged_df['就诊人数'] = merged_df['就诊人数'].fillna(0)

# 检查是否存在多余的合并键
if 'key_0' in merged_df.columns:
    merged_df.drop(columns=['key_0'], inplace=True)

# 保存合并后的数据
output_path = r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\气象与就诊人数.csv"
merged_df.to_csv(output_path, index=False)

print(f"合并后的数据已保存到 {output_path}")