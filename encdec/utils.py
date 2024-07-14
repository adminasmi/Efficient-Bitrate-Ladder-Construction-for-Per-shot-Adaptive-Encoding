""" 1. convert video format """
import os
import subprocess

def convertfmt(
        input_path,
        output_path,
        width,
        height,
        input_fmt="yuv420p",
        output_fmt="yuv420p10le"
):
    cmd = [
        "ffmpeg",
        "-y",
        "-s", f"{width}x{height}",
        "-pix_fmt", input_fmt,
        "-i", input_path,
        "-c:v", "rawvideo",
        "-pix_fmt", output_fmt,
        output_path
    ]
    try:
        subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error Conversion: {e}")


def countJobs(jobName):
    fileHandle = os.popen(f"ps -e | grep {jobName} | wc -l")
    return int(fileHandle.read())
