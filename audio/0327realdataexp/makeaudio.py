import soundfile as sf
import numpy as np
#根据录制的两条语音进行测试 #每个说话人取10s的语音长度 aux也取10s
def getoriaudio():
    spk1_w_path = "projects/debug/0327realdata/B401_chen_45_0327_6ch.wav"
    spk2_m_path = "projects/debug/0327realdata/B401_xlx_180_0327_6ch.wav"
    spk1_w,fs = sf.read(spk1_w_path, dtype="float32")
    spk2_m, fs = sf.read(spk2_m_path, dtype="float32")
    spk1_w = spk1_w[:,1:5]
    spk2_m = spk2_m[:,1:5]
    spk1_w_source = spk1_w[:10*fs, :]
    spk2_m_source = spk2_m[:10*fs, :]
    spk1_aux = spk1_w[-10*fs:, 0]
    spk2_aux = spk2_m[-10*fs:, 0]
    sf.write("projects/debug/0327realdata/spk1w_source.wav", spk1_w_source, fs)
    sf.write("projects/debug/0327realdata/spk2m_source.wav", spk2_m_source, fs)
    sf.write("projects/debug/0327realdata/spk1_aux.wav", spk1_aux, fs)
    sf.write("projects/debug/0327realdata/spk2_aux.wav", spk2_aux, fs)

def generratenewaux():
    spk1_w_path = "projects/debug/0327realdata/B401_chen_45_0327_6ch.wav"
    spk2_m_path = "projects/debug/0327realdata/B401_xlx_180_0327_6ch.wav"
    spk1_w,fs = sf.read(spk1_w_path, dtype="float32")
    spk2_m, fs = sf.read(spk2_m_path, dtype="float32")
    spk1_w = spk1_w[:,1:5]
    spk2_m = spk2_m[:,1:5]
    spk1_w_source = spk1_w[:10*fs, :]
    spk2_m_source = spk2_m[:10*fs, :]
    spk1_aux = spk1_w[20*fs:30*fs:, 0]
    spk2_aux = spk2_m[-10*fs:, 0]
    sf.write("projects/debug/0327realdata/spk1_newaux.wav", spk1_aux, fs)

def makemixaudio():
    w1_con,fs = sf.read("projects/debug/0327realdata/spk1w_source.wav", dtype="float32")
    w2_con,fs = sf.read("projects/debug/0327realdata/spk2m_source.wav", dtype="float32")
    mix_len = w1_con.shape[0]
    mix_ov100 = w1_con[:mix_len,:]+w2_con[:mix_len, :]
    sf.write("projects/debug/0327realdataexp/ov100_mix.wav", mix_ov100, samplerate=16000)
    sf.write("projects/debug/0327realdataexp/ov100_s1.wav", w1_con[:mix_len], samplerate=16000)
    #sf.write("projects/debug/0327realdataexp/ov100_s1aux.wav", w1_aux, samplerate=16000)
    sf.write("projects/debug/0327realdataexp/ov100_s2.wav", w2_con[:mix_len], samplerate=16000)
    #sf.write("projects/debug/0327realdataexp/ov100_s2aux.wav", w2_aux, samplerate=16000)
    
    # ov50 # 假设重叠部分为3.5s
    # w1_con = w1_con[:mix_len, :]
    # w2_con = w2_con[:mix_len, :]
    # w1_con = np.concatenate((np.random.random((mix_len//3, 4))*1e-8, w1_con), axis=0)
    # w2_con = np.concatenate((w2_con, np.random.random((mix_len//3, 4))*1e-8), axis=0)
    # mixov50 = w1_con+w2_con
    # sf.write("projects/debug/0327realdataexp/ov50_mix.wav", mixov50, samplerate=16000)
    # sf.write("projects/debug/0327realdataexp/ov50_s1.wav", w1_con, samplerate=16000)
    # #sf.write("projects/debug/0327realdataexp/ov50_s1aux.wav", w1_aux, samplerate=16000)
    # sf.write("projects/debug/0327realdataexp/ov50_s2.wav", w2_con, samplerate=16000)
    # #sf.write("projects/debug/0327realdataexp/ov50_s2aux.wav", w2_aux, samplerate=16000)
    
    # ov20 # 假设重叠部分为3.5s
    # w1_con = w1_con[:mix_len, :]
    # w2_con = w2_con[:mix_len, :]
    # w1_con = np.concatenate((np.random.random((mix_len//3*2, 4))*1e-8, w1_con), axis=0)
    # w2_con = np.concatenate((w2_con, np.random.random((mix_len//3*2, 4))*1e-8), axis=0)
    # mixov20 = w1_con+w2_con
    # sf.write("projects/debug/0327realdataexp/ov20_mix.wav", mixov20, samplerate=16000)
    # sf.write("projects/debug/0327realdataexp/ov20_s1.wav", w1_con, samplerate=16000)
    # #sf.write("projects/debug/0327realdataexp/ov50_s1aux.wav", w1_aux, samplerate=16000)
    # sf.write("projects/debug/0327realdataexp/ov20_s2.wav", w2_con, samplerate=16000)
    #sf.write("projects/debug/0327realdataexp/ov50_s2aux.wav", w2_aux, samplerate=16000)
    
    
    print(111)

if __name__=="__main__":
    #makemixaudio()
    generratenewaux()