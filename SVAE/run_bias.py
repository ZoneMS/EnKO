import os, sys, json
import time, shutil
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.utils
from torch.autograd import Variable
from util_bias import *



def main():
    date = "210730"
    number_of_date = "1"
    result_dir = os.path.join("bias_results", "{}_{}".format(date, number_of_date))
    
    f = open("config_bias.json", "r")
    config = json.load(f)
    f.close()
    
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(os.path.join(result_dir, "config_bias.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    dim_state_range = [int(i) for i in config["dim_state_range"]]
    dim_obs_range = [int(i) for i in config["dim_obs_range"]]
    
    number_of_results = 2 + (2+1+2) + 7
    transforms = list(config["transforms"].values())
    for i in range(len(transforms)):
        if transforms[i]=="square-tanh":
            number_of_results += 2
    
    system_list = ["fivo", "fivor", "enko", "iwae", "simple"]
    results = np.zeros([dim_state_range[1] - dim_state_range[0], dim_obs_range[1] - dim_obs_range[0], number_of_results, len(system_list)])
    for i, Dz in enumerate(range(dim_state_range[0], dim_state_range[1])):
        for j, Dx in enumerate(range(dim_obs_range[0], dim_obs_range[1])):
            start_time = time.time()
            results[i,j], _ = compute_bias_and_variance(Dz, Dx, config=config)
            print("Dz={},Dx={}:{} sec".format(Dz, Dx, time.time() - start_time))
            np.save(os.path.join(result_dir, "results.npy"), results[:i+1,:j+1])
    
    
if __name__ == "__main__":
    main()