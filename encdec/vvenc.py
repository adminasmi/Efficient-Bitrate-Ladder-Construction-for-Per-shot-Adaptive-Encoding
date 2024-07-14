import time
import os, re
from tqdm import tqdm
from encdec.utils import countJobs

sizes = ["2160P", "432P"]

size_map = {
    "2160P": "3840x2160",
    "1080P": "1920x1080",
    "720P": "1280x720",
    "540P": "960x540",
    "432P": "768x432",
    "360P": "640x360",
}
fps = 30

yuv_root = "/hdd/YoutubeUGC/scenes/yuv"
vvencapp = "/home/zhaoy/vvenc/bin/release-static/vvencapp"

rlt_root = "/hdd/YoutubeUGC/enc_rlts/vvenc"

qps = list(range(22, 57, 5))
presets = ["faster", "medium", "slower"]

for size in sizes:
    yuv_dir = os.path.join(yuv_root, size)

    for seq in tqdm(list(filter(lambda x: x.endswith(".yuv"), os.listdir(yuv_dir)))):
        # Lecture-2655_1080P_scene6.yuv
        seq_path = os.path.join(yuv_dir, seq)
        seq_name = seq.split("_")[0]
        scene_id = re.search(r"scene(\d+)", seq)[1]

        rlt_dir = os.path.join(rlt_root, size, seq_name)

        for qp in qps:
            for preset in presets:
                os.makedirs(f"{rlt_dir}/log", exist_ok=True)
                os.makedirs(f"{rlt_dir}/bin", exist_ok=True)

                log = os.path.join(rlt_dir, "log", seq.replace(".yuv", f"_qp{qp}_{preset}.log"))
                bin = os.path.join(rlt_dir, "bin", seq.replace(".yuv", f"_qp{qp}_{preset}.bin"))

                cmd = (f"{vvencapp} --size {size_map[size]} --preset {preset} --qp {qp} --fps {fps} --format yuv420 "
                       f"--input {seq_path} --passes 1 --output {bin} --threads 8 > {log} &")
                os.system(cmd)

                while countJobs("vvencapp") > 110:
                    time.sleep(0.5)