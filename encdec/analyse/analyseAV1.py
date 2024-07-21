size_map = {
    "2160P": "3840x2160",
    "1080P": "1920x1080",
    "720P": "1280x720",
    "540P": "960x540",
    "360P": "640x360",
}
import sys
sys.path.append("../../")

enc_root  = "/hdd/YoutubeUGC/enc_rlts/svtav1"
orig_root = "/hdd/YoutubeUGC/scenes"        # orig: Lecture-2513_1080P_scene7.yuv

vmaf_dir = "/home/zhaoy/vmaf"
sizes    = ["360P", "540P", "720P", "1080P"]

import os
import re
import time
import pandas as pd
from tqdm import tqdm
from encdec.utils import countJobs
from metrics import getPSNR, getSSIM, getVMAF, get_vvencInfo, calPSNR, calSSIM


def getEncInfo(save_dir="/home/zhaoy/asset-fastCAE/results/svtav1/tables", get_psnr=True, get_ssim=True, get_vmaf=True):
    allseqInfo = []
    for size in sizes:
        seqs = os.listdir(f"{enc_root}/{size}")
        for seq in tqdm(seqs, desc=f"size {size}"):
            seq_dir = os.path.join(enc_root, size, seq)
            rec_dir = os.path.join(seq_dir, "rec")
            metrics_dir = os.path.join(seq_dir, "metrics")

            for rec_yuv in os.listdir(rec_dir):
                seq_name = rec_yuv.split("_")[0]
                scene_id = re.search(r"scene(\d+)", rec_yuv)[1]
                qp = re.search("qp(\d+)", rec_yuv)[1]
                preset = rec_yuv.split(".")[0].split("_")[-1]

                enc_info = get_vvencInfo(os.path.join(seq_dir, "stat", rec_yuv.replace("yuv", "stat")), read_psnr=False)
                bitrate, nframes = enc_info[0], enc_info[1]

                try:
                    psnr = getPSNR(os.path.join(metrics_dir, "psnr", rec_yuv.replace("yuv", "txt")))  if get_psnr else -1
                    ssim = getSSIM(os.path.join(metrics_dir, "ssim", rec_yuv.replace("yuv", "txt")))  if get_ssim else -1
                    vmaf = getVMAF(os.path.join(metrics_dir, "vmaf", rec_yuv.replace("yuv", "json"))) if get_vmaf else -1
                    allseqInfo.append([seq_name, scene_id, qp, preset, size, nframes, bitrate, psnr, ssim, vmaf])
                except Exception as e:
                    print(f"Error {e} ({rec_yuv})")

    allseqInfoDf = pd.DataFrame(allseqInfo, columns=["seqName", "sceneId", "qp", "preset", "size", "nframes", "bitrate", "psnr", "ssim", "vmaf"])

    os.makedirs(save_dir, exist_ok=True)
    allseqInfoDf.to_csv(f"{save_dir}/encInfo.csv", index=False)


def cal_PSNR_SSIM():
    for size in sizes:
        seqs = os.listdir(f"{enc_root}/{size}")
        width, height = size_map[size].split("x")[0], size_map[size].split("x")[1]

        for seq in seqs:
            seq_dir = os.path.join(f"{enc_root}/{size}", seq)
            rec_dir = os.path.join(seq_dir, "rec")

            metrics_dir = os.path.join(seq_dir, "metrics")
            os.makedirs(f"{metrics_dir}/psnr", exist_ok=True)
            os.makedirs(f"{metrics_dir}/ssim", exist_ok=True)

            for rec_yuv in tqdm(os.listdir(rec_dir), desc=f"size:{size}, seq:{seq}"):
                # rec:  Lecture-003a_1080P_scene0_qp22_faster.yuv
                orig_path = os.path.join(orig_root, "yuv420p", size, rec_yuv.split("_qp")[0] + ".yuv")
                rec_path = os.path.join(rec_dir, rec_yuv)
                assert os.path.exists(orig_path) and os.path.exists(rec_path)

                calPSNR(
                    orig_path, rec_path, psnr_dir=os.path.join(metrics_dir, "psnr"),
                    orig_fmt="yuv420p", rec_fmt="yuv420p10le", height=height, width=width
                )
                calSSIM(
                    orig_path, rec_path, ssim_dir=os.path.join(metrics_dir, "ssim"),
                    orig_fmt="yuv420p", rec_fmt="yuv420p10le", height=height, width=width
                )
                while countJobs("ffmpeg") > 300:
                    time.sleep(0.5)


if __name__ == '__main__':
    cal_PSNR_SSIM()
    # getEncInfo()