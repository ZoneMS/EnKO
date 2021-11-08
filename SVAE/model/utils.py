import numpy as np
import math, subprocess
from torch import nn

def overlap_data_augmentation(data, window=300, overlap=0.5):
    T, Dx = data.shape
    n_sample = math.ceil((T-window)/(window*(1-overlap)))+1
    new_data = np.zeros((n_sample, window, Dx))
    #t_data = np.zeros(n_sample)
    for i in range(n_sample-1):
        new_data[i] = data[int(i*window*(1-overlap)):int(i*window*(1-overlap))+window]
        #t_data[i] = int(i*window*(1-overlap))
    new_data[-1] = data[-window:]
    #t_data[-1] = T-window
    return new_data


def calculate_conv_size(current, kernel, stride, padding):
    c = current
    for (k,s,p) in zip(kernel, stride, padding):
        c = math.floor((c + 2*p - k) / s + 1)
    return c


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class UnFlatten(nn.Module):
    def __init__(self,w,h):
        super().__init__()
        self.w = w
        self.h = h
    def forward(self, input):
        nc = input[0].numel()//(self.w*self.h)
        return input.view(input.size(0), nc, self.h, self.w)
    
    
    
def get_model(Dx, config, device):
    model_name = config["data"]["model"]

    if model_name=="VRNN":
        from model.vrnn import VRNN
        model =  VRNN(Dx, config, device).to(device)
    elif model_name=="MINN":
        from model.minn import MINN
        model =  MINN(Dx, config, device).to(device)
    elif model_name=="STORN":
        from model.storn import STORN
        model =  STORN(Dx, config, device).to(device)
    elif model_name=="SRNN":
        from model.srnn import SRNN
        model =  SRNN(Dx, config, device).to(device)
    elif model_name=="AESMC":
        from model.aesmc import AESMC
        model =  AESMC(Dx, config, device).to(device)
    elif model_name=="SVO":
        from model.svo import SVO
        model =  SVO(Dx, config, device).to(device)
    elif model_name=="SVO-II":
        from model.svo2 import SVO2
        model =  SVO2(Dx, config, device).to(device)
    elif model_name=="MINN-SVO":
        from model.minn3 import MINN3
        model =  MINN3(Dx, config, device).to(device)
    elif model_name=="PSVO":
        from model.psvo import PSVO
        model =  PSVO(Dx, config, device).to(device)
 
    return model



class Identity(nn.Module):
    def forward(self, input):
        return input



def get_nonlinear_fn(name="linear"):
    if name in ["Identity", "identity", "Linear", "linear"]:
        nonlinear_fn = Identity
    elif name in ["ReLU", "relu"]:
        nonlinear_fn = nn.ReLU
    elif name in ["Sigmoid", "sigmoid"]:
        nonlinear_fn = nn.Sigmoid
    elif name in ["Softmax", "softmax"]:
        nonlinear_fn = nn.Softmax
    elif name in ["Softplus", "softplus"]:
        nonlinear_fn = nn.Softplus
    elif name in ["Tanh", "tanh"]:
        nonlinear_fn = nn.Tanh
    return nonlinear_fn



def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=("index", "uuid", "name", "timestamp", "memory.total", "memory.free", "memory.used", "utilization.gpu", "utilization.memory"), no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]