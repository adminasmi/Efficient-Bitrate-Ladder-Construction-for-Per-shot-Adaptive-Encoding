import os
import re
from tqdm import tqdm

yuv_root = "/hdd/CTC/scenes/yuv420p/orig"
rlt_root = "/hdd/CTC/scenes/enc_rlts/svtav1"
av1enc   = "/home/zhaoy/AV1-Optimization/Bin/Release/SvtAv1EncApp"

qps = [22, 27, 32, 37]
size = "540P"
preset = 3

for type in ["lfr", "hfr"]:
    yuv_dir = os.path.join(yuv_root, size, type)
    for seq in tqdm(list(filter(lambda x: x.endswith(".yuv"), os.listdir(yuv_dir)))):
        # MarketPlace_1920x1080_60fps_10bit_420.yuv     Cactus_1920x1080_50.yuv
        seq_path = os.path.join(yuv_dir, seq)
        seq_name = seq.split("_")[0]
        fps = re.search(r"\d+x\d+_(\d+)", seq)[1]
        
        rlt_dir = os.path.join(rlt_root, size, type, seq.split(".")[0])
        os.makedirs(f"{rlt_dir}/log", exist_ok=True)
        os.makedirs(f"{rlt_dir}/bin", exist_ok=True)
        os.makedirs(f"{rlt_dir}/rec", exist_ok=True)
        
        for qp in qps:
            bin = os.path.join(rlt_dir, "bin", seq.replace(".yuv", f"_qp{qp}_preset{preset}.bin"))
            log = os.path.join(rlt_dir, "log", seq.replace(".yuv", f"_qp{qp}_preset{preset}.log"))
            rec = os.path.join(rlt_dir, "rec", seq.replace(".yuv", f"_qp{qp}_preset{preset}.yuv"))
            
            os.chdir(os.path.join(rlt_dir, "rec"))
            
            cmd = f"{av1enc} -i {seq_path} -w 960 -h 540 --tune 2 --preset 3 --fps {fps} --rc 0 --qp {qp} --enable-qm 1 --keyint 256 --hierarchical-levels 5 --enable-stat-report 1 --stat-file {log} -b {bin} -o {rec} &"
            os.system(cmd)