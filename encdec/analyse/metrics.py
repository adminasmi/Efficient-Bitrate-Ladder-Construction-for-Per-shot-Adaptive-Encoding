""" 一些指标计算的函数 """
import os
import re
import json
import subprocess
from collections import deque


# 0. read enc info from vvenc log
def get_vvencInfo(logpath, read_psnr=False):
    with open(logpath, "r") as f:
        lastlines = deque(f, 10)

        for line in lastlines:
            if " a " in line:
                parts = line.split()
                bitrate = float(parts[4])
                psnr_y = float(parts[5])
                psnr_u = float(parts[6])
                psnr_v = float(parts[7])
                psnr   = (psnr_y * 6 + psnr_u + psnr_v) / 8.0

            if "vvencapp" in line:
                nframes = int(re.search(r"encoded Frames (\d+)", line)[1])

    if read_psnr:
        return [bitrate, psnr_y, psnr_u, psnr_v, psnr, nframes]
    else:
        return [bitrate, nframes]


def get_av1Info(logpath):
    # Picture Number:  375     QP:   30  [ PSNR-Y: 20.85 dB,  PSNR-U: 31.60 dB,       PSNR-V: 31.61 dB,       MSE-Y: 534.52,  MSE-U: 44.98,   MSE-V: 44.88,   SSIM-Y: 0.27658,        SSIM-U: 0.66819,        SSIM-V: 0.68039 ]          2042 bytes
    psnr_y, psnr_u, psnr_v = 0, 0, 0
    ssim_y, ssim_u, ssim_v = 0, 0, 0
    nframes = 0
    bitrate = 0
    with open(logpath, "r") as f:
        lines = f.readlines()
        for line in lines:
            pattern = re.compile(
                r"PSNR-Y:\s*(?P<psnr_y>[\d.]+) dB,\s*"
                r"PSNR-U:\s*(?P<psnr_u>[\d.]+) dB,\s*"
                r"PSNR-V:\s*(?P<psnr_v>[\d.]+) dB,\s*"
                r"SSIM-Y:\s*(?P<ssim_y>[\d.]+),\s*"
                r"SSIM-U:\s*(?P<ssim_u>[\d.]+),\s*"
                r"SSIM-V:\s*(?P<ssim_v>[\d.]+)*"
                r"(?P<size>\d+) bytes"
            )
            match = pattern.match(line)

            if match:
                data = match.groupdict()
                psnr_y  += float(data['psnr_y'])
                psnr_u  += float(data['psnr_u'])
                psnr_v  += float(data['psnr_v'])
                ssim_y  += float(data['ssim_y'])
                ssim_u  += float(data['ssim_u'])
                ssim_v  += float(data['ssim_v'])
                bitrate += int(data['size'])
                nframes += 1

    psnr = (6 * psnr_y + psnr_u + psnr_v) / (8 * nframes)
    ssim = (6 * ssim_y + ssim_u + ssim_v) / (8 * nframes)

    return  [bitrate, nframes, psnr, ssim]


# 1. vmaf
def calVMAF(
        orig_path,
        rec_path,
        vmaf_dir,
        test_script = "/home/zhaoy/vmaf/python/vmaf/script/run_vmaf.py",
        out_fmt  = "json",
        out_file = None,
        pix_fmt  = "yuv420p10le",
        height = 270,
        width  = 480
):
    cmd = ["python", test_script, pix_fmt, f"{width}", f"{height}", orig_path, rec_path, "--out-fmt", out_fmt]

    env = os.environ.copy()
    env["PYTHONPATH"] = "python"

    try:
        rlt = subprocess.run(cmd, cwd=vmaf_dir, env=env, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running VMAF script: {e}")
        return -1

    if out_file and out_fmt == "json":
        data = json.loads(rlt.stdout)
        with open(out_file, "w") as f:
            json.dump(data, f, indent=4)
    else:
        print("STDOUT:", rlt.stdout)


# 2. psnr & ssim -> use ffmpeg
def calPSNR(
        orig_path,
        rec_path,
        psnr_dir,
        orig_fmt="yuv420p",
        rec_fmt="yuv420p10le",
        height = 270,
        width = 480,
        cover_prev = True,
        scale_width  = 1920,
        scale_height = 1080
):
    psnr_log = os.path.join(psnr_dir, os.path.split(rec_path)[-1].replace(".yuv", ".txt"))
    if os.path.exists(psnr_log):
        if cover_prev:
            os.system(f"rm -f {psnr_log}")
        else:
            return

    cmd = (
       f"ffmpeg -y "
       f"-s {width}x{height} -pix_fmt {rec_fmt}  -i {rec_path} "
       f"-s {width}x{height} -pix_fmt {orig_fmt} -i {orig_path} "
       f"-lavfi '"
       f"[0:v]scale=w={scale_width}:h={scale_height}:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,setpts=PTS-STARTPTS[reference];"
       f"[1:v]scale=w={scale_width}:h={scale_height}:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,setpts=PTS-STARTPTS[distorted];"
       f"[distorted][reference]psnr=stats_file={psnr_log}' -f null - &"
   )
    os.system(cmd)


def calSSIM(
        orig_path,
        rec_path,
        ssim_dir,
        orig_fmt="yuv420p",
        rec_fmt="yuv420p10le",
        height=270,
        width=480,
        cover_prev=True,
        scale_width  = 1920,
        scale_height = 1080
):
    ssim_log = os.path.join(ssim_dir, os.path.split(rec_path)[-1].replace(".yuv", ".txt"))
    if os.path.exists(ssim_log):
        if cover_prev:
            os.system(f"rm -f {ssim_log}")
        else:
            return

    cmd = (
       f"ffmpeg -y "
       f"-s {width}x{height} -pix_fmt {rec_fmt}  -i {rec_path} "
       f"-s {width}x{height} -pix_fmt {orig_fmt} -i {orig_path} "
       f"-lavfi '"
       f"[0:v]scale=w={scale_width}:h={scale_height}:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,setpts=PTS-STARTPTS[reference];"
       f"[1:v]scale=w={scale_width}:h={scale_height}:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,setpts=PTS-STARTPTS[distorted];"
       f"[distorted][reference]ssim=stats_file={ssim_log}' -f null - &"
    )
    os.system(cmd)


# 3. read psnr and ssim from .txt files
def getPSNR(log_path):
    # n:161 mse_avg:1.78 mse_y:2.52 mse_u:0.20 mse_v:0.40 psnr_avg:57.69 psnr_y:56.18 psnr_u:67.26 psnr_v:64.18
    cnt = 0
    psnr_avg = 0
    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "inf" not in line:
            psnr_avg += float(re.search(r"psnr_avg:(\d+.\d+)", line)[1])
            cnt      += 1
    psnr_avg /= cnt

    return psnr_avg


def getSSIM(log_path):
    # n:1 Y:0.997794 U:0.997933 V:0.998095 All:0.997867 (26.710411)
    cnt = 0
    ssim_avg = 0
    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        ssim_avg += float(re.search(r"All:(\d.\d+)", line)[1])
        cnt      += 1
    ssim_avg /= cnt

    return ssim_avg


def getVMAF(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)
    vmaf = data["aggregate"]["VMAF_score"]

    return vmaf