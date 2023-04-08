## 对测试语音进行剪裁
import soundfile as sf

# audio_path = "projects/debug/0408realdata/meeting3_chen_xu_0180_0408_6ch.wav"
# audio, fs = sf.read(audio_path, dtype="float32")
# audio = audio[48*fs:int(-2.8*fs), 1:5]
# sf.write("projects/debug/0408realdataexp/sample1_chenxu_0180_mix.wav", audio, samplerate=fs)
audio_path = "projects/debug/0408realdata/meeting3_chen_xu_90180_0408_6ch.wav"
audio, fs = sf.read(audio_path, dtype="float32")
audio = audio[int(1.2*fs):int(95*fs), 1:5]
sf.write("projects/debug/0408realdataexp/sample2_chenxu_90180_mix.wav", audio, samplerate=fs)
