import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
## 画语谱图
def plotfun(path):

    #path = "projects/debug/0404systhsisaudio/offline_4ch_TSE_spk1m_spk2w_ov20_mix_TSE_s2.wav"

    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    data, fs = librosa.load(path, sr=None) 
    L = len(data)

    print('Time:', L / fs)

    data = data/max(abs(data))
    data = data+np.random.rand(L)*1e-8 
    #0.025s
    framelength = 0.025
    #NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    print("NFFT:", framesize)

    #画语谱图
    plt.figure(figsize=(6,4))
    plt.subplot(2,1,1)
    plt.specgram(data, NFFT=framesize, Fs=fs, window=np.hanning(M=framesize))
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.subplot(2,1,2)
    t = np.linspace(0,L, L)/fs
    plt.plot(t, data)
    savedir = os.path.dirname(path)
    savename = os.path.basename(path).split(".")[0]+".png"
    savename = os.path.join(savedir, savename)
    plt.savefig(savename)

## 将文件夹下的所有音频都画出语谱图
def ffmpeg_MP3ToWav(input_path):
    # 提取input_path路径下所有文件名
    filename = os.listdir(input_path)
    for file in filename:
        if file[-4:] ==".wav":
            path1 = input_path + "/" + file
            plotfun(path1)

input_path = "projects/debug/0404systhsisaudio"            
ffmpeg_MP3ToWav(input_path)
