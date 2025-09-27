import pandas as pd
from pandas import DataFrame
import numpy as np

primitive_data = pd.read_csv("data-sample-and-filter.csv", encoding='GBK')
print("原始数据形状:", primitive_data.shape)

primitive_data_1 = primitive_data.dropna(how='any')
print("删除空行后形状:", primitive_data_1.shape)

data_before_filter = primitive_data_1
data_after_filter_1 = data_before_filter.loc[data_before_filter["traffic"] != 0]
data_after_filter_2 = data_after_filter_1.loc[data_after_filter_1["from_level"] == '一般节点']
print("过滤后数据形状:", data_after_filter_2.shape)

data_before_sample = data_after_filter_2
columns = data_before_sample.columns

# 1. 加权抽样
weight_sample = data_before_sample.copy()
weight_sample['weight'] = 0

for i in weight_sample.index:
    if weight_sample.at[i, 'to_level'] == '一般节点':
        weight = 1
    else:
        weight = 5
    weight_sample.at[i, 'weight'] = weight

weight_sample_finish = weight_sample.sample(n=50, weights='weight')
weight_sample_finish = weight_sample_finish[columns]
weight_sample_finish['抽样方法'] = '加权抽样'
print("加权抽样完成，样本数:", len(weight_sample_finish))

# 2. 随机抽样
random_sample_finish = data_before_sample.sample(n=50)
random_sample_finish = random_sample_finish[columns]
random_sample_finish['抽样方法'] = '随机抽样'
print("随机抽样完成，样本数:", len(random_sample_finish))

# 3. 分层抽样
ybjd = data_before_sample.loc[data_before_sample['to_level'] == '一般节点']
wlhx = data_before_sample.loc[data_before_sample['to_level'] == '网络核心']
after_sample = pd.concat([ybjd.sample(17), wlhx.sample(33)])
after_sample = after_sample[columns]
after_sample['抽样方法'] = '分层抽样'
print("分层抽样完成，样本数:", len(after_sample))

# 4. 系统抽样
def systematic_sampling(data, n):
    N = len(data)
    k = N // n  
    start = np.random.randint(0, k) 
    indices = [start + i * k for i in range(n) if (start + i * k) < N]
    return data.iloc[indices]

systematic_sample = systematic_sampling(data_before_sample, 50)
systematic_sample = systematic_sample[columns]
systematic_sample['抽样方法'] = '系统抽样'
print("系统抽样完成，样本数:", len(systematic_sample))

# 5. 整群抽样
def cluster_sampling(data, n_clusters, cluster_col='to_level'):
    clusters = data[cluster_col].unique()
    selected_clusters = np.random.choice(clusters, size=n_clusters, replace=False)
    cluster_sample = data[data[cluster_col].isin(selected_clusters)]
    if len(cluster_sample) > 50:
        cluster_sample = cluster_sample.sample(n=50)
    return cluster_sample

cluster_sample = cluster_sampling(data_before_sample, n_clusters=2, cluster_col='to_level')
cluster_sample = cluster_sample[columns]
cluster_sample['抽样方法'] = '整群抽样'
print("整群抽样完成，样本数:", len(cluster_sample))


all_samples = pd.concat([
    weight_sample_finish,
    random_sample_finish,
    after_sample,
    systematic_sample,
    cluster_sample
], ignore_index=True)

print("\n所有抽样方法合并后总样本数:", len(all_samples))

all_samples.insert(0, '样本编号', range(1, len(all_samples) + 1))

output_filename = "五种抽样方法结果汇总.csv"
all_samples.to_csv(output_filename, index=False, encoding='GBK')
print(f"\n结果已保存到文件: {output_filename}")

sample_counts = all_samples['抽样方法'].value_counts()
print("\n各抽样方法样本数量统计:")
print(sample_counts)
