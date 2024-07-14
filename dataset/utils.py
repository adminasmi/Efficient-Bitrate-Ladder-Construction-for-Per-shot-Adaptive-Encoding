import re
import os
import random
import pandas as pd

from tqdm import tqdm
from scenedetect import VideoManager, SceneManager, ContentDetector


def countJobs(jobName):
    """ 检测某个任务的数量 """
    fileHandle = os.popen(f"ps -e | grep {jobName} | wc -l")
    return int(fileHandle.read())


# 去除音频
def rmAudio(origPath, outDir):
    fileName = os.path.split(origPath)[-1]
    newfileName = f"an_{fileName}"
    os.system(f"ffmpeg -i {origPath} -c:v copy -an {os.path.join(outDir, newfileName)} &")


# ------------------ fmt translation ------------------
def transMp4ToYuv(orig_seq_path, seq_size, out_dir, pixfmt="yuv420p"):
    outyuvfile = os.path.split(orig_seq_path)[-1].split(".")[0] + ".yuv"
    outyuvpath = os.path.join(out_dir, outyuvfile)

    cmd = f"ffmpeg -i {orig_seq_path} -s {seq_size} -pix_fmt {pixfmt} -loglevel error {outyuvpath} &"
    # print(cmd)
    os.system(cmd)


def transYuvToMp4(src_path, rlt_dir, size, fps, pixfmt="yuv420p"):
    outmp4file = os.path.split(src_path)[-1].split(".")[0] + ".mp4"
    outmp4path = os.path.join(rlt_dir, outmp4file)

    cmd = f"ffmpeg -y -s {size} -r {fps} -pix_fmt {pixfmt} -i {src_path} -c:v libx265 -x265-params log-level=0 -crf 18 -loglevel error {outmp4path} 2>&1 > transtomp4.log &"
    os.system(cmd)
    # print(cmd)


def trans265ToMP4(orig_bin_path, out_dir):
    seq_name = (os.path.split(orig_bin_path)[-1]).split(".")[0] + ".mp4"

    cmd = f"ffmpeg -i {orig_bin_path} -c copy {os.path.join(out_dir, seq_name)} &"
    print(cmd)
    os.system(cmd)


def splitScene(videoPath, sceneDir, threshold=15.0, codec="libx265"):
    """ video : .mp4 """
    os.makedirs(sceneDir, exist_ok=True)
    videoManager = VideoManager([videoPath])

    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector(threshold=threshold))

    # 场景检测
    videoManager.set_downscale_factor()     # speed-up
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager, show_progress=True)

    # 获取场景列表（以帧为单位）
    sceneList = sceneManager.get_scene_list(videoManager.get_base_timecode())

    videofile = os.path.split(videoPath)[-1]
    seqName   = videofile.split(".")[0]

    # 输出每个场景的开始和结束时间
    for idx, scene in enumerate(sceneList):
        startFrame = scene[0].get_frames()
        endFrame   = scene[1].get_frames()

        scenePath = os.path.join(sceneDir, f"{seqName}_scene{idx}.mkv")

        # 用 ffmpeg 分割场景
        cmd = f"/usr/bin/ffmpeg -y -i {videoPath} -vf 'select=between(n\,{startFrame}\,{endFrame})' -c:v {codec} -x265-params log-level=0 -loglevel error {scenePath}"
        os.system(cmd)

# 把序列拆成一帧一帧（.png）
def splitSeqframes(seqPath, rltRootDir):
    seq_name = os.path.split(seqPath)[-1].split(".")[0]
    rlt_dir  = os.path.join(rltRootDir, seq_name)

    os.makedirs(rlt_dir, exist_ok=True)

    cmd = f"ffmpeg -i {seqPath} -loglevel error {os.path.join(rlt_dir, 'im%d.png')} &"
    # print(cmd)
    os.system(cmd)


# 排序生成的 .png 图片 (im1.png, im2.png, ...)
def sortFrames(root_dir):
    frames = os.listdir(root_dir)

    # 提取文件名中的 frameId
    def extract_nums(file):
        if os.path.isdir(file):
            file = os.path.split(file)[-1]

        match = re.search(r"\d+", file)
        if match:
            return int(match.group())

        return 0

    # 根据 frameId 为他们排序
    sorted_frames = sorted(frames, key=extract_nums)


    return sorted_frames


# 这里只做空域下采样，时域下采样通过丢帧实现
def rescaleMp4Size(orig_seq_path, scaleW, scaleH, scale_seq_dir, seq_name, codec="libx265"):
    """ rescale .mp4/.mkv sequence to a new size """
    # 'scaleSeqDir' is a directory, seqfile's format: '<class>-<ID>_<reso>.mp4'

    scale_seq_path = os.path.join(scale_seq_dir, f"{seq_name}_{scaleH}P.mkv")

    scaleH = (scaleH + 1) if (scaleH % 2 != 0) else scaleH
    scaleW = (scaleW + 1) if (scaleW % 2 != 0) else scaleW
    cmd = f"ffmpeg -y -i {orig_seq_path} -vf scale={scaleW}:{scaleH} -c:v {codec} -crf 0 -max_muxing_queue_size 4096 {scale_seq_path} &"
    print(cmd)
    # os.system(cmd)


size_map = {1080: "1920x1080", 720: "1280x720", 540: "960x540", 480: "854x480", 360: "640x360", 270: "480x270"}

def rescaleYuvSize(orig_seq_path, scaleW, scaleH, scale_seq_dir, seq_name):
    """ rescale .yuv sequence into a new size """
    origH = re.search(r"(\d+)P", seq_name)[1]

    scaleH = (scaleH + 1) if (scaleH % 2 != 0) else scaleH
    scaleW = (scaleW + 1) if (scaleW % 2 != 0) else scaleW

    scale_seq_path = os.path.join(scale_seq_dir, seq_name.replace(f"{origH}", f"{scaleH}"))
    cmd = f"ffmpeg -y -s {size_map[int(origH)]} -pix_fmt yuv420p -i {orig_seq_path} -vf scale={scaleW}x{scaleH} -pix_fmt yuv420p {scale_seq_path}"

    print(cmd)
    os.system(cmd)


def genFlistTxt(frame_dir, sel_frames, flist_path):
    """
    generate filelist.txt for frame-combining
    :param frame_dir:  directory of source frames
    :param sel_frames: list of selected frames (containing their ID number)
    :param flist_path: path of filelist.txt
    """
    with open(flist_path, "w") as flist:
        for frameId in sel_frames:
            flist.write(f"file \'{frame_dir}/im{frameId}.png\'\n")


'''
`flist_txt` is a .txt file with each row starting with the word `file`. e.g.:
    file './1.png'
    file './3.png'
    file './5.png'
'''
def combineFrames(flist_txt, dst_seq_path):
    """ combine a series of frames into a single sequence """
    cmd = f"ffmpeg -f concat -safe 0 -i {flist_txt} -pix_fmt yuv420p {dst_seq_path} &"
    print(cmd)
    os.system(cmd)


""" 查看当前目录下，各个子目录分别包含了多少文件 """
def cntSubdirlen(curr_dir):
    rows = []

    for subdir in os.listdir(curr_dir):
        subdir = os.path.join(curr_dir, subdir)

        if os.path.isdir(subdir):
            rows.append([subdir, len(os.listdir(subdir))])

    sublens = pd.DataFrame(rows, columns=["seq_subdir", "frame_number"])

    return sublens


def genkeys(data_root_dir):
    keys = set()
    for seq in tqdm(os.listdir(data_root_dir)):
        seq_dir = os.path.join(data_root_dir, seq)

        for group in os.listdir(seq_dir):
            if os.path.isdir(os.path.join(seq_dir, group)):
                keys.add(f"{seq}_{group}")

    return keys


def septraintest(cache_keys, septxt_dir, test_ratio=0.15):
    keys = list(cache_keys["keys"])
    random.shuffle(keys)

    train_size = int(len(keys) * (1 - test_ratio))
    test_size = len(keys) - train_size

    for mode in ["train", "fast_test", "medium_test", "slow_test", "test"]:
        with open(f"{septxt_dir}/sep_{mode}list.txt", "w", encoding="utf-8") as f:
            for item in eval(f"{mode}_set"):
                sub_dir = item.split("_")[-1]
                dir = "_".join(item.split("_")[:-1])
                f.write(f"{dir}/{sub_dir}\n")

        print(f"len of {mode}_set: {len(eval(f'{mode}_set'))}")


def regroup(seq, num_frames, src_dir, dst_dir, group_size=5, verbose=False):
    """ if `inplace` is TRUE then directly operate in `srcDir` """
    # `src_dir` 对应一个 scene (i.e. df 中的一行), `dst_dir` 也对应一个 scene
    src_file = os.path.split(src_dir)[-1] if os.path.isdir(src_dir) else src_dir

    if num_frames < group_size:  # 如果小于 5 帧，直接放弃这个片段
        print(f"{src_file} only contains {num_frames} frames, skip it.")

        if os.path.exists(dst_dir):
            os.system(f"rm -rf {dst_dir}")

    else:
        num_groups = num_frames // group_size
        num_resiframes = num_frames % group_size

        if verbose:
            print(f"{seq}: Having {num_groups} groups. {num_resiframes} leftover frames.")

        frame_ids = list(range(1, num_frames + 1))
        frame_groups = [frame_ids[i:i + group_size] for i in range(0, num_frames, group_size)]

        groupId = 1
        for group in frame_groups:
            if len(group) == group_size:
                # g 是一个 frame_group, 包含了几帧的 id
                group_dir = os.path.join(dst_dir, f"{groupId:04d}")
                os.makedirs(group_dir, exist_ok=True)

                for frameId in group:
                    src_frame_path = os.path.join(src_dir, f"im{frameId}.png")
                    dst_frame_path = os.path.join(group_dir, f"im{(frameId - 1) % group_size + 1}.png")

                    os.system(f"cp {src_frame_path} {dst_frame_path}")

            groupId += 1