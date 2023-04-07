## 分离模型测试
## 提取模型测试
import sys
sys.path.append("/home/chen/chen/code_chen/lightweight_ss_4mic")
import os
import json
import torch
from asteroid import torch_utils
import numpy as np

## 导入onnx推理库
import onnx
import onnxruntime as ort
import yaml
import soundfile as sf
import matplotlib.pyplot as plt

## load model
def load_best_model(model, exp_dir):
    # Create the model from recipe-local function
    try:
        # Last best model summary
        with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        # Get last checkpoint
        all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt]
        all_ckpt.sort(key=lambda x:x[1])
        best_model_path = os.path.join(exp_dir, 'checkpoints', all_ckpt[-1][0])
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

def loadtestaudio(mix_path, targetpath, aux_path):
    mix,_ = sf.read(mix_path, dtype="float32")
    s1,_ = sf.read(targetpath, dtype="float32")
    aux,_ = sf.read(aux_path, dtype="float32")
    return mix, s1, aux

def testmodeloffline4ch(mixpath, targetpath, auxpath):
    ##模型测试 #offline  4 channel test
    conffile = "exp/FFCnoise/batchsize16lr0.002/conf.yml"
    from FFC.mymodel import make_model_and_optimizer

    with open(conffile) as f:
        conf = yaml.safe_load(f)
    model, _ = make_model_and_optimizer(conf) ## 加载模型
    model = load_best_model(model, conf["main_args"]['exp_dir'])  
    
    mix_np, s1_np, aux_np = loadtestaudio(mixpath, targetpath,auxpath)
    
    mix_tr = torch.from_numpy(mix_np.astype(np.float32)).unsqueeze(1).permute(1,2,0)
    aux_tr = torch.from_numpy(aux_np.astype(np.float32)).unsqueeze(0)
    
    est_s1 = model(mix_tr, aux_tr)
    est_s1_np = est_s1.squeeze().detach().numpy() 
    
    expname ="offline_4ch_SEP_"+os.path.basename(mixpath).split('.')[0]+"_SEP_"+os.path.basename(targetpath).split('_')[-1]
    expname = expname.split(".")[0]
    sf.write(os.path.join("projects/debug/0404systhsisaudio", expname+".wav"), est_s1_np.transpose(1,0), samplerate=16000)
    print("show extract results")
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    plt.plot(mix_np[:,0])
    plt.title("mix audio")
    plt.subplot(3,1,2)
    plt.plot(s1_np[:,0])
    plt.title("ideal audio")
    plt.subplot(3,1,3)
    plt.plot(est_s1_np[0,:])
    plt.title("extarct audio")
    plt.show()
    plt.savefig(os.path.join("projects/debug/0404systhsisaudio", expname+".png"))
    print(111)