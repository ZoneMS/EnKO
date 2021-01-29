import os, sys, json
import time, shutil, copy
import math
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from run_svae import main

    
def gs_main():
    f = open("config.json", "r")
    config = json.load(f)
    f.close()
    
    parallel_on = True
    number_of_date = ""
    
    gs_dict = {"system":["EnKO", "FIVO", "IWAE"]}
    gs_key = list(gs_dict.keys())
    gs_length = len(gs_dict)
    
    topic_name = ""
    for topic in ["system", "model", "data_name"]:
        if not (topic in gs_key):
            if not (topic in ["system"] and config ["data"][topic] is None):
                topic_name += config["data"][topic] + "_"
    
    dir_name = "{}gs{}/".format(topic_name, number_of_date)
    config_list, name_list = [], []
    
    def generate_queue(old_config, name, depth):
        key = gs_key[depth]

        for i, value in enumerate(gs_dict[key]):
            new_name = name
            new_config = copy.deepcopy(old_config)
            for ckey in config.keys():
                if key in config[ckey].keys():
                    new_config[ckey][key] = value
                    abbrev_list = key.split("_")
                    for abbrev in abbrev_list:
                        new_name += abbrev[0]
                    new_name += str(value)
                    break

            if depth+1 < gs_length:
                generate_queue(new_config, new_name, depth+1)
            else:
                config_list.append(new_config)
                name_list.append(new_name)
                #main(config, new_name)
    
    generate_queue(config, dir_name, 0)
    
    if parallel_on:
        p = mp.Pool(min(mp.cpu_count(), len(name_list)))
        p.starmap(main, zip(config_list, name_list))
        p.close()
    else:
        for config, name in zip(config_list, name_list):
            main(config, name)
    

    
            
            
if __name__ == "__main__":
    gs_main()