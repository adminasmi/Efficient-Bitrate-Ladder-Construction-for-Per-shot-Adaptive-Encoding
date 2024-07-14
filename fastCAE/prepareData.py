""" 生成所有数据集 """
import re, os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

table_dir = "/home/zhaoy/asset-fastCAE/results/vvenc/tables"
data_dir  = "/home/zhaoy/asset-fastCAE/dataset/vvenc"

targets = ["bitrate", "log2bitrate", "psnr", "log2psnr", "ssim", "log2ssim", "vmaf", "log2vmaf"]
funcs = ["linear", "power", "quadratic2", "quadratic3"]

def extract_floats(s, pattern = r"[-+]?\d*\.\d+e?[-+]?\d*"):
    matches = re.findall(pattern, s)
    floats = [float(num) for num in matches]
    return pd.Series(floats)


corr_df = pd.read_csv(f"{table_dir}/corrs.csv")
grouped = corr_df.groupby(["seqName"], as_index=False)
groups = list(grouped.groups.keys())

# train_seqs and test_seqs
train_set, test_set = train_test_split(groups, test_size=0.1, random_state=42)

for func in funcs:
    os.makedirs(f"{data_dir}/corr_{func}", exist_ok=True)

    for target in tqdm(targets, desc=f"{func}"):
        params = ["p1", "p2", "p3"] if func in ["quadratic3"] else ["p1", "p2"]
        df = corr_df[(corr_df["target"] == target) & (corr_df["func"] == func)].reset_index(drop=True)
        df[params] = df["popt"].apply(extract_floats)

        os.makedirs(f"{data_dir}/corr_{func}", exist_ok=True)
        df.to_csv(f"{data_dir}/corr_{func}/corr_{target}.csv", index=False)

        # 划分训练集与测试集
        train_df = df[df["seqName"].isin(train_set)].reset_index(drop=True)
        test_df = df[df["seqName"].isin(test_set)].reset_index(drop=True)

        train_df.to_csv(f"{data_dir}/corr_{func}/corr_{target}_train.csv", index=False)
        test_df.to_csv(f"{data_dir}/corr_{func}/corr_{target}_test.csv", index=False)