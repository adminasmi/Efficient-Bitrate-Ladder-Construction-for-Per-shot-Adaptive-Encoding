import time
import os
from tqdm import tqdm
from utils import countJobs

sizes = ["1080P", "720P", "540P", "360P"]

size_map = {
    "2160P": "3840x2160",
    "1080P": "1920x1080",
    "720P": "1280x720",
    "540P": "960x540",
    "360P": "640x360",
}
fps = 30

yuv_root = "/hdd/YoutubeUGC/scenes/yuv420p"
x265 = "/home/zhaoy/x265/build/linux/x265"

rlt_root = "/hdd/YoutubeUGC/enc_rlts/x265"

qps = list(range(22, 57, 5))
presets = ["faster", "medium", "slower"]

for size in sizes:
    yuv_dir = os.path.join(yuv_root, size)

    for seq in tqdm(list(filter(lambda x: x.endswith(".yuv"), os.listdir(yuv_dir)))):
        # Lecture-2655_1080P_scene6.yuv
        seq_path = os.path.join(yuv_dir, seq)
        seq_name = seq.split("_")[0]

        rlt_dir = os.path.join(rlt_root, size, seq_name)

        for qp in qps:
            for preset in presets:
                os.makedirs(f"{rlt_dir}/log", exist_ok=True)
                os.makedirs(f"{rlt_dir}/bin", exist_ok=True)
                os.makedirs(f"{rlt_dir}/rec", exist_ok=True)

                csv = os.path.join(rlt_dir, "log", seq.replace(".yuv", f"_qp{qp}_{preset}.csv"))
                bin = os.path.join(rlt_dir, "bin", seq.replace(".yuv", f"_qp{qp}_{preset}.bin"))
                rec = os.path.join(rlt_dir, "rec", seq.replace(".yuv", f"_qp{qp}_{preset}.yuv"))

                cmd = (f"{x265} --input-res {size_map[size]} --preset {preset} --qp {qp} --fps 30 --psnr --ssim --tune psnr --input {seq_path} --output {bin} --recon {rec} --csv-log-level 2 --csv {csv} --log-level info &")
                os.system(cmd)

                while countJobs("x265") > 110:
                    time.sleep(0.5)