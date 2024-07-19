import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor
from utils import func_linear, func_pw2, func_quad2, func_exp

def _process_single(args):
    seqName, sceneId, preset, size, df, funcs, targets = args
    rows = []

    x = df["qp"]
    for target in targets:
        y = df[target]
        for func in funcs:
            try:
                popt, pconv = curve_fit(f=func, xdata=x, ydata=y, full_output=False, maxfev=3000)
                pred_value = func(x, *popt)
                abs_error = pred_value - y
                r2 = 1.0 - (np.var(abs_error) / np.var(y))
                row = [seqName, sceneId, preset, size, tuple(popt),
                       func.__name__.replace("func_", "").replace("pw2", "power").replace("quad", "quadratic"), target,
                       r2] + y.tolist()[:5]
                rows.append(row)

            except Exception as e:
                print(f"{e} (seqName={seqName}, sceneId={sceneId}, size={size}, func={func.__name__})")
                continue

    return rows


funcs = [func_linear, func_pw2, func_quad2, func_exp]
targets = ["log2bitrate", "log2psnr", "log2ssim", "log2vmaf", "bitrate", "psnr", "ssim", "vmaf"]

table_dir = "/home/zhaoy/asset-fastCAE/results/vvenc/tables"
enc_df = pd.read_csv(f"{table_dir}/interp_encInfo.csv")

rows = []
tasks = []
for (seqName, sceneId, preset, size), group in enc_df.groupby(["seqName", "sceneId", "preset", "size"]):
    group = group.reset_index(drop=True)
    tasks.append((seqName, sceneId, preset, size, group, funcs, targets))

with ThreadPoolExecutor(max_workers=24) as executor:
    results = list(tqdm(executor.map(_process_single, tasks), total=len(tasks)))

for result in results:
    rows.extend(result)

cols = ["seqName", "sceneId", "preset", "size", "popt", "func", "target", "r2"] + [f"y{i}" for i in range(5)]
r2_scores_df = pd.DataFrame(rows, columns=cols)

r2_scores_df.to_csv(f"{table_dir}/corrs.csv", index=False)
