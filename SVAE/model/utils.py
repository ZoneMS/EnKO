import numpy as np
import math
from torch import nn
    
def get_model(Dx, config, device):
    model_name = config["data"]["model"]

    if model_name=="VRNN":
        from model.vrnn import VRNN
        model =  VRNN(Dx, config, device).to(device)
    elif model_name=="SRNN":
        from model.srnn import SRNN
        model =  SRNN(Dx, config, device).to(device)
    elif model_name=="AESMC":
        from model.aesmc import AESMC
        model =  AESMC(Dx, config, device).to(device)
    elif model_name=="SVO":
        from model.svo import SVO
        model =  SVO(Dx, config, device).to(device)
    elif model_name=="PSVO":
        from model.psvo import PSVO
        model =  PSVO(Dx, config, device).to(device)
 
    return model



class Identity(nn.Module):
    def forward(self, input):
        return input



def get_nonlinear_fn(name="linear"):
    if name in ["Identity", "identity", "Linear", "linear"]:
        nonlinear_fn = Identity()
    elif name in ["ReLU", "relu"]:
        nonlinear_fn = nn.ReLU()
    elif name in ["Sigmoid", "sigmoid"]:
        nonlinear_fn = nn.Sigmoid()
    elif name in ["Softmax", "softmax"]:
        nonlinear_fn = nn.Softmax()
    elif name in ["Softplus", "softplus"]:
        nonlinear_fn = nn.Softplus()
    elif name in ["Tanh", "tanh"]:
        nonlinear_fn = nn.Tanh()
    return nonlinear_fn