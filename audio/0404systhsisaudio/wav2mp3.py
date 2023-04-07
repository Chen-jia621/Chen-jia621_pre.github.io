# coding=UTF-8
import os
import subprocess


def ffmpeg_MP3ToWav(input_path, output_path):
    # 提取input_path路径下所有文件名
    filename = os.listdir(input_path)
    for file in filename:
        if file[-4:] ==".wav":
            path1 = input_path + "/" + file
            path2 = output_path + "/" + os.path.splitext(file)[0]
            cmd = "ffmpeg -i " + path1 + " " + path2 + ".mp3" #将input_path路径下所有音频文件转为.mp3文件
            subprocess.call(cmd, shell=True)

input_path = "projects/debug/0404systhsisaudio"
output_path = "projects/debug/0404systhsisresultmp3"
ffmpeg_MP3ToWav(input_path, output_path)
