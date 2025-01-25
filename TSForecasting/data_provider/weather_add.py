import pandas as pd
import os

# 定义文件路径
file_paths = [
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2017.csv",
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2018.csv",
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2019.csv",
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2020.csv",
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2021.csv",
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2022.csv",
    r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2023.csv"
]

# 初始化一个空的DataFrame
combined_df = pd.DataFrame()

# 逐个读取文件并合并
for file_path in file_paths:
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 保存合并后的数据到新的CSV文件
output_path = r"E:\浏览器下载地址\宁海疾控食源性疾病\TimeNet\dataset\weather\2017-2023.csv"
combined_df.to_csv(output_path, index=False)

print(f"数据已合并并保存到 {output_path}")