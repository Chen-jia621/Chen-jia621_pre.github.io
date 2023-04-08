## 真实数据提取实验
## 提取模型测试
import sys
sys.path.append("/home/chen/chen/code_chen/lightweight_ss_4mic")
import os
import json
import torch
from asteroid import torch_utils
import numpy as np
from src.mvdr_util import MVDR

## 导入onnx推理库
import onnx
import onnxruntime as ort
import yaml
import soundfile as sf
import matplotlib.pyplot as plt

## load model
def load_best_model(model, exp_dir):
    # Create the model from recipe-local function

    # Last best model summary
    with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)

    print( 'LOADING from ',best_model_path)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location='cpu')
    for k in list(checkpoint['state_dict'].keys()):
        if('loss_func' in k):
            del checkpoint['state_dict'][k]
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], model)
    model = model.eval()
    return model

## 读取测试数据
def loadtestaudio(mix_path, aux_path):
    mix,_ = sf.read(mix_path, dtype="float32")
    aux,_ = sf.read(aux_path, dtype="float32")
    return mix, aux

def testmodeloffline4ch(mixpath, auxpath):
    ##模型测试 #offline  4 channel test
    conffile = "exp/FFC_TSEnoise/batchsize8lr0.002/conf.yml"
    from FFC_TSE.mymodel import make_model_and_optimizer

    with open(conffile) as f:
        conf = yaml.safe_load(f)
    model, _ = make_model_and_optimizer(conf) ## 加载模型
    model = load_best_model(model, conf["main_args"]['exp_dir'])  
    mvdr = MVDR(False)
    mix_np, aux_np = loadtestaudio(mixpath, auxpath)
    
    mix_tr = torch.from_numpy(mix_np.astype(np.float32)).unsqueeze(1).permute(1,2,0)
    aux_tr = torch.from_numpy(aux_np.astype(np.float32)).unsqueeze(0)
    
    
    #mix_tr = mix_tr[:,:10*16000]## 10s
    est_s1 = model(mix_tr, aux_tr)
    
    
    ## do MVDR #看看MVDR是否可以去除或者降低残余噪声
    ## MVDR bscn
    #est_s1 = mvdr(mix_tr, est_s1.unsqueeze(1))
    
    
    est_s1_np = est_s1.squeeze().detach().numpy() 
    
    
    
    expname ="offline_4ch_TSE_"+os.path.basename(mixpath).split('.')[0]+"_TSE_"+os.path.basename(auxpath).split('_')[0]
    expname = expname.split(".")[0]
    
    sf.write(os.path.join(os.path.dirname(mixpath), expname+".wav"), est_s1_np[0,:], samplerate=16000)
    print("show extract results")
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(mix_np[:,0])
    plt.title("mix audio")
    plt.subplot(2,1,2)
    plt.plot(est_s1_np[0,:])
    plt.title("extarct audio")
    plt.show()
    os.makedirs(os.path.dirname(mixpath)+"/fig", exist_ok=True) ## 生成目标路径
    plt.savefig(os.path.join(os.path.dirname(mixpath)+"/fig", expname+".png"))
    print(111)

def testmodelonline2ch(mixpath, auxpath): ## 
    ##模型测试 #offline  4 channel test
    conffile = "exp/causalTSEmymodel2ch001_causal/batchsize8lr0.002/conf.yml"
    from causal_TSE.mymodel2ch_001_causal import make_model_and_optimizer

    with open(conffile) as f:
        conf = yaml.safe_load(f)
    model, _ = make_model_and_optimizer(conf) ## 加载模型
    model = load_best_model(model, conf["main_args"]['exp_dir'])   
    
    mix_np, aux_np = loadtestaudio(mixpath, auxpath)
    
    mix_tr = torch.from_numpy(mix_np.astype(np.float32)).unsqueeze(1).permute(1,2,0)
    aux_tr = torch.from_numpy(aux_np.astype(np.float32)).unsqueeze(0)
    
    
    
    est_s1 = model(mix_tr, aux_tr)
    est_s1_np = est_s1.squeeze().detach().numpy() 
    
    expname ="online_2ch_TSE_"+os.path.basename(mixpath).split('.')[0]+"_TSE_"+os.path.basename(auxpath).split('_')[0]
    expname = expname.split(".")[0]
    
    sf.write(os.path.join(os.path.dirname(mixpath), expname+".wav"), est_s1_np, samplerate=16000)
    print("show extract results")
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(mix_np[:,0])
    plt.title("mix audio")
    plt.subplot(2,1,2)
    plt.plot(est_s1_np)
    plt.title("extarct audio")
    plt.show()
    os.makedirs(os.path.dirname(mixpath)+"/fig", exist_ok=True) ## 生成目标路径
    plt.savefig(os.path.join(os.path.dirname(mixpath)+"/fig", expname+".png"))
    print(111)


def testmodelonline2chvad(mixpath, auxpath): ## 效果有点差
     ##模型测试 #offline  2 channel test vad
    conffile = "exp/causalTSEmymodel2ch001_causal_vad_noiseloss/batchsize8lr0.002/conf.yml"
    from causal_TSE.mymodel2ch_001_causal_vad import make_model_and_optimizer

    with open(conffile) as f:
        conf = yaml.safe_load(f)
    model, _ = make_model_and_optimizer(conf) ## 加载模型
    model = load_best_model(model, conf["main_args"]['exp_dir'])  
    
    mix_np, aux_np = loadtestaudio(mixpath, auxpath)
    
    mix_tr = torch.from_numpy(mix_np.astype(np.float32)).unsqueeze(1).permute(1,2,0)
    aux_tr = torch.from_numpy(aux_np.astype(np.float32)).unsqueeze(0)
    
    
    
    est_s1, est_s1_vad = model(mix_tr, aux_tr)
    est_s1_np = est_s1.squeeze().detach().numpy() 
    est_s1_vad = est_s1_vad.squeeze().detach().numpy()
    
    expname ="online_2ch_TSEVAD_"+os.path.basename(mixpath).split('.')[0]+"_TSE_"+os.path.basename(auxpath).split('_')[0]
    expname = expname.split(".")[0]
    
    sf.write(os.path.join(os.path.dirname(mixpath), expname+".wav"), est_s1_np, samplerate=16000)
    print("show extract results")
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    plt.plot(mix_np[:,0])
    plt.title("mix audio")
    plt.subplot(3,1,2)
    plt.plot(est_s1_np)
    plt.title("extarct audio")
    plt.subplot(3,1,3)
    plt.plot(est_s1_vad)
    plt.title("extarct audio vad")
    plt.show()
    os.makedirs(os.path.dirname(mixpath)+"/fig", exist_ok=True) ## 生成目标路径
    plt.savefig(os.path.join(os.path.dirname(mixpath)+"/fig", expname+".png"))
    print(111)
    


if __name__=="__main__":
    mixname = "projects/debug/0408realdataexp/sample1_chenxu_0180_mix.wav"
    #s1_name = "projects/debug/0327realdataexp/ov20_s1.wav"
    auxname = "projects/debug/0327realdata/spk2_aux.wav"
    #testmodelonline2chvad(mixname, auxname)
    testmodeloffline4ch(mixname, auxname)
    #testmodeloffline4ch(mixname, s1_name, auxname)## 出现了错误提取的情况 离线模型反而出现了 虽然离线模型的提取概率的sdr高一些 # 换一个aux可以解决这个问题
    #testmodelonline2ch(mixname, s1_name, auxname)
    pass