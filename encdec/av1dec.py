import os
from tqdm import tqdm

sizes = ["360P", "540P", "720P", "1080P"]
size_map = {
    "2160P": "3840x2160",
    "1080P": "1920x1080",
    "720P": "1280x720",
    "540P": "960x540",
    "360P": "640x360",
}
av1dec = "/home/zhaoy/SVT-AV1/Bin/Release/SvtAv1DecApp"

def decYoutubeUGC():
    root = "/hdd/YoutubeUGC/enc_rlts/svtav1"
    for size in sizes:
        width, height = size_map[size].split("x")[0], size_map[size].split("x")[1]
        sub_root = os.path.join(root, size)

        seqs = os.listdir(sub_root)
        for seq in tqdm(seqs):
            bin_dir = os.path.join(sub_root, seq, "bin")
            rec_dir = os.path.join(sub_root, seq, "rec")
            os.makedirs(rec_dir, exist_ok=True)

            for bin_file in list(filter(lambda x: x.endswith(".bin"), os.listdir(bin_dir))):
                # Lecture-003a_1080P_scene1_qp42_11.bin
                bin_path = os.path.join(bin_dir, bin_file)
                rec_path = os.path.join(rec_dir, bin_file.replace(".bin", ".yuv"))

                if os.path.exists(rec_path):
                    os.system(f"rm -f {rec_path}")

                cmd = f"{av1dec} -i {bin_path} -w {width} -h {height} -o {rec_path}"
                os.system(cmd)


def decCTC():
    root = "/hdd/CTC/enc_rlts/svtav1"
    classes = ["B", "C", "D", "E", "F"]
    for c in classes:
        for size in sizes:
            width, height = size_map[size].split("x")[0], size_map[size].split("x")[1]
            sub_root = os.path.join(root, c, size)
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

                    cmd = f"{av1dec} -i {bin_path} -w {width} -h {height} -o {rec_path}"
                    os.system(cmd)

if __name__ == '__main__':
    decCTC()