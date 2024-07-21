""" Encoding/Decoding based on SVT-AV1 """
import os, re
from tqdm import tqdm

sizes = ["540P"]
size_map = {
    "2160P": "3840x2160",
    "1080P": "1920x1080",
    "720P": "1280x720",
    "540P": "960x540",
    "360P": "640x360",
}
fps = 30


av1enc = "/home/zhaoy/SVT-AV1/Bin/Release/SvtAv1EncApp"
av1dec = "/home/zhaoy/SVT-AV1/Bin/Release/SvtAv1DecApp"

qps = list(range(22, 58, 3))
presets = [2, 5, 8, 11]


def encYoutubeUGC():
    yuv_root = "/hdd/YoutubeUGC/scenes/yuv420p"
    rlt_root = "/hdd/YoutubeUGC/enc_rlts/svtav1"
    for size in sizes:
        yuv_dir = os.path.join(yuv_root, size)
        width, height = size_map[size].split("x")[0], size_map[size].split("x")[1]

        for seq in tqdm(list(filter(lambda x: x.endswith(".yuv"), os.listdir(yuv_dir)))):
            # Lecture-2655_1080P_scene6.yuv
            seq_path = os.path.join(yuv_dir, seq)
            seq_name = seq.split("_")[0]

            rlt_dir = os.path.join(rlt_root, size, seq_name)

            for qp in qps:
                for preset in presets:
                    os.makedirs(f"{rlt_dir}/bin", exist_ok=True)
                    os.makedirs(f"{rlt_dir}/log", exist_ok=True)

                    bin  = os.path.join(rlt_dir, "bin", seq.replace(".yuv", f"_qp{qp}_{preset}.bin"))
                    stat = os.path.join(rlt_dir, "log", seq.replace(".yuv", f"_qp{qp}_{preset}.stat"))

                    if os.path.exists(bin) and os.path.exists(stat):
                        continue

                    cmd = f"{av1enc} -i {seq_path} -w {width} -h {height} --fps {fps} --qp {qp} --preset {preset} --pass 1 --enable-stat-report 1 --stat-file {stat} -b {bin}"
                    os.system(cmd)


classes = ["B", "C", "D", "E", "F"]
def encCTC():
    yuv_root = "/hdd/CTC/yuv420p"
    rlt_root = "/hdd/CTC/enc_rlts/svtav1"

    for c in classes:
        for size in sizes:
            yuv_dir = os.path.join(yuv_root, c, size)
            width, height = size_map[size].split("x")[0], size_map[size].split("x")[1]

            for seq in tqdm(list(filter(lambda x: x.endswith(".yuv"), os.listdir(yuv_dir)))):
                seq_path = os.path.join(yuv_dir, seq)
                seq_name = seq.split("_")[0]
                fps = re.search(r"\d+x\d+_(\d+)", seq)[1]

                rlt_dir = os.path.join(rlt_root, c, size, seq_name)

                for qp in qps:
                    for preset in presets:
                        os.makedirs(f"{rlt_dir}/bin", exist_ok=True)
                        os.makedirs(f"{rlt_dir}/log", exist_ok=True)

                        bin = os.path.join(rlt_dir, "bin", seq.replace(".yuv", f"_qp{qp}_{preset}.bin"))
                        stat = os.path.join(rlt_dir, "log", seq.replace(".yuv", f"_qp{qp}_{preset}.stat"))

                        if os.path.exists(bin) and os.path.exists(stat):
                            continue

                        cmd = f"{av1enc} -i {seq_path} -w {width} -h {height} --fps {fps} --qp {qp} --preset {preset} --pass 1 --enable-stat-report 1 --stat-file {stat} -b {bin}"
                        os.system(cmd)


if __name__ == '__main__':
    encCTC()
