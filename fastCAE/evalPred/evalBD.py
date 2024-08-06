import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import ConvexHull

import pandas as pd
pd.set_option('display.max_rows', 6)

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',  # 使用衬线字体
    'font.serif': ['Times New Roman'],  # 指定 Times New Roman 字体
    'font.size': 12,
    'text.usetex': False,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.linewidth': 1,
    'axes.linewidth': 1
})
sns.set_style("whitegrid")
flierprops = dict(marker='o', markersize=2, linestyle='none')

cols = ["seqName", "sceneId", "regressor", "func", "input", "preset", "size", "qp", "pred_target"]


def process_group(args):
    group, seqName, sceneId, preset, regressor, inputs, distortion = args

    distortion = distortion.lower()
    assert distortion.lower() in ["psnr", "log2psnr", "ssim", "log2ssim", "vmaf", "log2vmaf"]
    is_log_distortion = ("log2" in distortion.lower())
    distortion_type = distortion.split("log2")[-1]

    bitrate_df = group[
        (group["target"] == "log2bitrate") &
        (group["regressor"] == regressor) &
        (group["func"] == "quadratic2") &  # log2bitrate 固定使用 quadratic2
        (group["input"] == inputs)
        ][cols + ["log2bitrate", "bitrate"]].reset_index(drop=True)
    bitrate_df = bitrate_df.drop(columns=["func"])
    bitrate_df = bitrate_df.rename(columns={"pred_target": "pred_log2bitrate"})

    if bitrate_df.empty: return

    for func in group[(group["target"] == distortion) & (group["regressor"] == regressor) & (group["input"] == inputs)]["func"].unique():
        distortion_df = group[
            (group["target"] == distortion) &
            (group["regressor"] == regressor) &
            (group["func"] == func) &
            (group["input"] == inputs)
            ][cols + [f"log2{distortion_type}", distortion_type]].reset_index(drop=True)
        distortion_df = distortion_df.rename(columns={"pred_target": f"pred_{distortion}"})

        group_rd = pd.merge(distortion_df, bitrate_df, how="inner")
        group_rd["pred_bitrate"] = 2 ** group_rd["pred_log2bitrate"]
        if is_log_distortion:
            group_rd[f"pred_{distortion_type}"] = 2 ** group_rd[f"pred_{distortion}"]

        inputs = inputs.replace("(", "").replace(")", "")

        """ 1. 得到凸包 """
        points = group_rd[["bitrate", distortion_type]].values
        actual_convex = ConvexHull(points)

        actual_hull_points = points[actual_convex.vertices]
        actual_hull_points = actual_hull_points[np.argsort(actual_hull_points[:, 0])]

        points = group_rd[["pred_bitrate", f"pred_{distortion_type}"]].values
        pred_convex = ConvexHull(points)
        pred_hull_points = points[pred_convex.vertices]
        pred_hull_points = pred_hull_points[np.argsort(pred_hull_points[:, 0])]

        # 保存凸包所在各行
        pred_convex_df = group_rd.iloc[pred_convex.vertices].reset_index(drop=True)
        actual_convex_df = group_rd.iloc[actual_convex.vertices].reset_index(drop=True)
        pred_convex_df["convex"] = "pred"
        actual_convex_df["convex"] = "actual"

        # 保存 fixed QP 对应各行 (QP = 27, preset == medium / 5)
        fixed_convex_df = group_rd[group_rd["qp"] == 27 & group_rd["preset"].isin(["faster", "5"])].reset_index(drop=True)
        fixed_convex_df["convex"] = "fixed"

        os.makedirs(f"{table_dir}/BDBR/rd-{distortion}/{seqName}-scene{sceneId}", exist_ok=True)

        convex_df = pd.concat([pred_convex_df, actual_convex_df], axis=0)
        convex_df = pd.concat([convex_df, fixed_convex_df], axis=0).reset_index(drop=True)
        convex_df.to_csv(f"{table_dir}/BDBR/rd-{distortion}/{seqName}-scene{sceneId}/{seqName}-scene{sceneId}_{regressor}_{func}_{preset}_{inputs}.csv", index=False)

        """ 2. 画图 """
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        sns.lineplot(ax=axes[0], data=group_rd, x="pred_bitrate", y=f"pred_{distortion_type}", hue="size", marker="o")
        axes[0].plot(pred_hull_points[:, 0], pred_hull_points[:, 1], linestyle='--', color='purple', marker="x", lw=2, alpha=0.6, label="predicted convex hull")
        axes[0].legend()
        axes[0].set_xlabel("Predicted Bitrate (kbps)")
        ylabel = "PSNR (dB)" if distortion_type == "psnr" else distortion_type.upper()
        axes[0].set_ylabel(f"Predicted {ylabel}")

        sns.lineplot(ax=axes[1], data=group_rd, x="bitrate", y=distortion_type, hue="size", marker="o")
        axes[1].plot(actual_hull_points[:, 0], actual_hull_points[:, 1], linestyle='--', color='purple', marker="x", lw=2, alpha=0.6, label="actual convex hull")
        axes[1].legend()
        axes[1].set_xlabel("Actual Bitrate (kbps)")
        axes[1].set_ylabel(f"Actual {ylabel}")

        x_min = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
        x_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

        axes[0].set_xlim(x_min, x_max)
        axes[1].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(y_min, y_max)

        plt.subplots_adjust(wspace=0.1)
        plt.tight_layout()

        os.makedirs(f"{fig_dir}/BDBR/rd-{distortion}/{seqName}-scene{sceneId}", exist_ok=True)
        plt.savefig(f"{fig_dir}/BDBR/rd-{distortion}/{seqName}-scene{sceneId}/{seqName}-scene{sceneId}_{regressor}_{func}_{preset}_{inputs}.pdf", format="pdf")
        plt.close()


def DEBUG(pred_df):
    group = pred_df[(pred_df["seqName"] == "Lecture-42c3") & (pred_df["sceneId"] == 0) & (pred_df["preset"] == "faster") & (pred_df["regressor"] == "Adam") & (pred_df["input"] == "faster")].reset_index(drop=True)
    process_group((group, "Lecture-42c3", 0, "faster", "Adam", "faster", "psnr"))


if __name__ == "__main__":
    table_dir = "/home/zhaoy/asset-fastCAE/results/vvenc/tables"
    fig_dir   = "/home/zhaoy/asset-fastCAE/results/vvenc/figs"

    pred_df = pd.read_csv(f"{table_dir}/predCurve/combined_preds.csv")
    pred_df = pred_df.drop(columns=["p1", "p2", "pred_p1", "pred_p2"])     # 不用看参数了
    if "p3" in pred_df.columns.tolist():
        pred_df = pred_df.drop(columns=["p3", "pred_p3"])

    # DEBUG(pred_df)
    grouped = pred_df.groupby(["seqName", "sceneId", "preset", "regressor", "input"], as_index=False)

    distortions = ["psnr", "log2psnr", "ssim", "log2ssim", "vmaf", "log2vmaf"]
    tasks = []
    for distortion in distortions:
        for (seqName, sceneId, preset, regressor, inputs), group in tqdm(grouped, desc=f"BD-Rate ({distortion})"):
            tasks.append((group, seqName, sceneId, preset, regressor, inputs, distortion))

    with ProcessPoolExecutor() as executor:
        executor.map(process_group, tasks)