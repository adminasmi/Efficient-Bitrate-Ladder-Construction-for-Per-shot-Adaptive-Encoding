import time
import os
from tqdm import tqdm

root = "/hdd/YoutubeUGC/enc_rlts/vvenc"
vvdecapp = "/home/zhaoy/vvdec/bin/release-static/vvdecapp"

# sizes = ["360P", "540P", "720P", "1080P"]

sizes = ["1080P"]

def countJobs(jobName):
    """ 检测某个任务的数量 """
    fileHandle = os.popen(f"ps -e | grep {jobName} | wc -l")
    return int(fileHandle.read())


for size in sizes:
    sub_root = os.path.join(root, size)
    
    seqs = os.listdir(sub_root)
    for seq in tqdm(seqs):
        bin_dir = os.path.join(sub_root, seq, "bin")
        rec_dir = os.path.join(sub_root, seq, "rec")
        os.makedirs(rec_dir, exist_ok=True)
        
        for bin_file in list(filter(lambda x: x.endswith(".bin"), os.listdir(bin_dir))):
            bin_path = os.path.join(bin_dir, bin_file)
            rec_path = os.path.join(rec_dir, bin_file.replace(".bin", ".yuv"))
            
            if os.path.exists(rec_path):
                os.system(f"rm -f {rec_path}")
                
            cmd = f"{vvdecapp} -b {bin_path} -o {rec_path} --threads 8 &"
            os.system(cmd)
            
            while countJobs("vvdecapp") > 120:
                time.sleep(0.5)
