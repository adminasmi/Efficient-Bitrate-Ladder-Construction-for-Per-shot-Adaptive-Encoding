size_map = {
    "2160P": "3840x2160",
    "1080P": "1920x1080",
    "720P": "1280x720",
    "540P": "960x540",
    "360P": "640x360",
}
import sys
sys.path.append("../../")

import os
import re
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from encdec.utils import countJobs
from metrics import getPSNR, getSSIM, getVMAF, calPSNR, calSSIM, calVMAF, get_av1Info

vmaf_dir = "/home/zhaoy/vmaf"
sizes = ["360P", "540P", "720P", "1080P"]


def getYoutubeEncInfo(save_dir="/home/zhaoy/asset-fastCAE/results/svtav1/tables"):
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

                enc_info = get_av1Info(os.path.join(seq_dir, "log", rec_yuv.replace("yuv", "stat")))
                bitrate, nframes = enc_info[0], enc_info[1]
                try:
                    psnr = getPSNR(os.path.join(metrics_dir, "psnr", rec_yuv.replace("yuv", "txt")))
                    ssim = getSSIM(os.path.join(metrics_dir, "ssim", rec_yuv.replace("yuv", "txt")))
                    vmaf = getVMAF(os.path.join(metrics_dir, "vmaf", rec_yuv.replace("yuv", "json")))
                    allseqInfo.append([seq_name, scene_id, qp, preset, size, nframes, bitrate, psnr, ssim, vmaf])
                except Exception as e:
                    print(f"Error {e} ({rec_yuv})")


    allseqInfoDf = pd.DataFrame(allseqInfo, columns=["seqName", "sceneId", "qp", "preset", "size", "nframes", "bitrate", "psnr", "ssim", "vmaf"])

    os.makedirs(save_dir, exist_ok=True)
    allseqInfoDf.to_csv(f"{save_dir}/encInfo.csv", index=False)


def calMetrics(cal_psnr=True, cal_ssim=True, cal_vmaf=True):
    for size in sizes:
        seqs = os.listdir(f"{enc_root}/{size}")
        width, height = size_map[size].split("x")[0], size_map[size].split("x")[1]

        # parallel calculation for VMAF
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for seq in tqdm(seqs):
                seq_dir = os.path.join(f"{enc_root}/{size}", seq)
                rec_dir = os.path.join(seq_dir, "rec")

                metrics_dir = os.path.join(seq_dir, "metrics")
                os.makedirs(f"{metrics_dir}/psnr", exist_ok=True)
                os.makedirs(f"{metrics_dir}/ssim", exist_ok=True)
                os.makedirs(f"{metrics_dir}/vmaf", exist_ok=True)

                for rec_yuv in os.listdir(rec_dir):
                    # rec:  Lecture-003a_1080P_scene0_qp22_faster.yuv
                    orig_path = os.path.join(orig_root, "yuv420p", size, rec_yuv.split("_qp")[0] + ".yuv")
                    rec_path = os.path.join(rec_dir, rec_yuv)
                    assert os.path.exists(orig_path) and os.path.exists(rec_path)

                    if cal_psnr:
                        calPSNR(
                            orig_path, rec_path, psnr_dir=os.path.join(metrics_dir, "psnr"),
                            orig_fmt="yuv420p", rec_fmt="yuv420p10le", height=height, width=width
                        )
                    if cal_ssim:
                        calSSIM(
                            orig_path, rec_path, ssim_dir=os.path.join(metrics_dir, "ssim"),
                            orig_fmt="yuv420p", rec_fmt="yuv420p10le", height=height, width=width
                        )
                    while countJobs("ffmpeg") > 300:
                        time.sleep(0.5)

                    if cal_vmaf:
                        futures.append(
                            executor.submit(
                                calVMAF,
                                os.path.join(orig_root, "yuv420p10le", size, rec_yuv.split("_qp")[0] + ".yuv"),
                                rec_path,
                                vmaf_dir,
                                "/home/zhaoy/vmaf/python/vmaf/script/run_vmaf.py",
                                "json",
                                os.path.join(f"{seq_dir}/metrics/vmaf", rec_yuv.replace(".yuv", ".json")),
                                "yuv420p10le",
                                height,
                                width
                            )
                        )

            for futures in tqdm(as_completed(futures), total=len(futures)):
                futures.result()


if __name__ == '__main__':
    enc_root = "/hdd/YoutubeUGC/enc_rlts/svtav1"
    orig_root = "/hdd/YoutubeUGC/scenes"  # orig: Lecture-2513_1080P_scene7.yuv

    calMetrics(cal_psnr=False, cal_ssim=False, cal_vmaf=True)
    # getEncInfo()