import os, sys, json
import time, shutil, copy
import subprocess

import math
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt 

import comet_ml
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from run_svae import main
from utils import get_gpu_info

    
def gs_main():
    f = open("config.json", "r")
    config = json.load(f)
    f.close()
    
    flatten_config = {}
    for key in config.keys():
        flatten_config.update(config[key])
    
    date = config["train"]["date"]
    number_of_date = config["train"]["number_of_date"]
    parallel_strategy_on = True
    direct_parse_strategy_on = True
    max_parallel_queues = min(6, mp.cpu_count())
    minimum_memory = 1500
    gpu_id = config["train"]["gpu"]
    #device = torch.device("cuda", config["train"]["gpu"])
    #torch.cuda.manual_seed(config["train"]["seed"])
    
    gs_dict = {"mix":{"system":["EnKO", "FIVO", "IWAE"], "loss_type":["sumprod", "prodsum", "sumprod"]}, "mix2":{"seed":[2, 3], "gpu":[1,2]}}
#     gs_dict = {"mix":{"system":["EnKO", "FIVO", "IWAE"], "loss_type":["sumprod", "prodsum", "sumprod"], "gpu":[0,1,2]},
#                "outer_scale":[1.0, 2.0]}
#     gs_dict = {"mix":{"system":["EnKO", "FIVO", "IWAE"], "gpu":[0,1,2]}}
#     gs_dict = {"mix":{"system":["EnKO", "FIVO", "IWAE"], "loss_type":["sumprod", "prodsum", "sumprod"]},
#                "n_particles":[8,16,32], "seed":[1,2,3]}
#     gs_dict = {"inflation_method":["RTPP", "RTPS"], "inflation_factor":[0.1, 0.2, 0.3], "n_particles":[8,16,32],
#               "mix":{"seed":[1,2,3], "gpu":[0,1,2]}}
#     gs_dict = {"inflation_method":["RTPP", "RTPS"], "seed":[1,2,3], "mix":{"inflation_factor":[0.1, 0.2, 0.3], "gpu":[0,1,2]}}
#     gs_dict = {"seed":[2,3], "mix":{"inflation_method":["RTPP", "RTPS"], "gpu":[1,2]}, "inflation_factor":[0.1, 0.2, 0.3]}
#     gs_dict = {"mix":{"inflation_factor":[0.1, 0.2, 0.3], "gpu":[0,1,2]}}
#     gs_dict = {"mix":{"loss_type":["prodsum", "sumprod"], "exclude_filtering_on":[False, True], "gpu":[0,1]}}
#     gs_dict = {"mix":{"system":["EnKO", "EnKO", "FIVO", "IWAE"],
#                       "loss_type":["sumprod", "prodsum", "prodsum", "sumprod"],
#                       "exclude_filtering_on":[True, False, True, True], "gpu":[1,2,1,2]}}
#     gs_dict = {"kld_penalty_weight":[0.1, 1.]}
#     gs_dict = {"mix":{"kld_penalty_weight":[0, 1.], "gpu":[1,2]}, "inflation_factor":[0.2, 0.3]}
#     gs_dict = {"mix":{"only_outer_learning_epochs":[0,10,0,10], "pretrain_epochs":[0,0,10,20]}}
#     gs_dict = {"seed":[1,2,3]}
    gs_key = list(gs_dict.keys()) # list of keys for grid search
    gs_length = len(gs_dict)
    gs_key2 = []
    for key in gs_key:
        # if dictionary has hierarchical structure, add hierarchical keys to gs_key2 list.
        if type(gs_dict[key])==list:
            gs_key2.append(key)
        elif type(gs_dict[key])==dict:
            gs_key2 += list(gs_dict[key].keys())
        
    
    topic_name = ""
    for topic in ["system", "outer_model", "model", "data_name"]:
        if not (topic in gs_key2):
            if not (topic in ["system", "outer_model"] and config ["data"][topic] is None):
                topic_name += config["data"][topic] + "_"
    
    dir_name = "{}_{}gs{}/".format(date, topic_name, number_of_date)
    name_list = []
    config_list = []
    if direct_parse_strategy_on:
        parse_list = []
        initial_parse = []
        for key in gs_key2:
            initial_parse += ["--{}".format(key), None]
        
        
    def generate_queue_flatten_config(old_config, name, depth):
        key = gs_key[depth]
        
        if type(gs_dict[key])==list:
            for i, value in enumerate(gs_dict[key]):
                new_name = name
                new_config = copy.deepcopy(old_config)
                new_config[key] = value
                abbrev_list = key.split("_")
                for abbrev in abbrev_list:
                    new_name += abbrev[0]
                new_name += str(value)
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, new_name, depth+1)
                else:
                    config_list.append(new_config)
                    name_list.append(new_name)
        elif type(gs_dict[key])==dict:
            interlocking_key = list(gs_dict[key].keys())
            min_length = 10
            for ikey in interlocking_key:
                min_length = len(gs_dict[key][ikey]) if len(gs_dict[key][ikey]) < min_length else min_length
            for i in range(min_length):
                new_name = name
                new_config = copy.deepcopy(old_config)
                for ikey in interlocking_key:
                    new_config[ikey] = gs_dict[key][ikey][i]
                    abbrev_list = ikey.split("_")
                    for abbrev in abbrev_list:
                        new_name += abbrev[0]
                    new_name += str(gs_dict[key][ikey][i])
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, new_name, depth+1)
                else:
                    config_list.append(new_config)
                    name_list.append(new_name)
        else:
            raise ValueError("elements must be a list type object or a dict type object")
            
    
    def flatten_config_to_parse(config):
        parse_list = []
        for key in config.keys():
            parse_list.append("--{}".format(key))
            if type(config[key])==list:
                parse_list += [str(value) for value in config[key]]
            else:
                parse_list.append(str(config[key]))
        return parse_list
    
    
    def generate_queue_parse(old_parse, name, depth, total_depth):
        key = gs_key[depth]
        
        if type(gs_dict[key])==list:
            for i, value in enumerate(gs_dict[key]):
                new_name = name
                new_parse = copy.deepcopy(old_parse)
                new_parse[2*total_depth+1] = str(value)
                abbrev_list = key.split("_")
                for abbrev in abbrev_list:
                    new_name += abbrev[0]
                new_name += str(value)
                if depth+1 < gs_length:
                    generate_queue_parse(new_parse, new_name, depth+1, total_depth+1)
                else:
                    parse_list.append(new_parse)
                    name_list.append(new_name)
        elif type(gs_dict[key])==dict:
            inner_keys = list(gs_dict[key].keys())
            # arrange length of inner list
            min_length = 10
            for ikey in inner_keys:
                min_length = len(gs_dict[key][ikey]) if len(gs_dict[key][ikey]) < min_length else min_length
            for i in range(min_length):
                new_name = name
                new_parse = copy.deepcopy(old_parse)
                for j, ikey in enumerate(inner_keys):
                    new_parse[2*(total_depth+j)+1] = str(gs_dict[key][ikey][i])
                    abbrev_list = ikey.split("_")
                    for abbrev in abbrev_list:
                        new_name += abbrev[0]
                    new_name += str(gs_dict[key][ikey][i])
                if depth+1 < gs_length:
                    generate_queue_parse(new_parse, new_name, depth+1, total_depth+len(inner_keys))
                else:
                    parse_list.append(new_parse)
                    name_list.append(new_name)
        else:
            raise ValueError("elements must be a list type object or a dict type object")
            
            
    def generate_queue_config(old_config, name, depth):
        key = gs_key[depth]
        
        if type(gs_dict[key])==list:
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
                    generate_queue_config(new_config, new_name, depth+1)
                else:
                    config_list.append(new_config)
                    name_list.append(new_name)
                    #main(config, new_name)
        elif type(gs_dict[key])==dict:
            interlocking_key = list(gs_dict[key].keys())
            min_length = 10
            for ikey in interlocking_key:
                min_length = len(gs_dict[key][ikey]) if len(gs_dict[key][ikey]) < min_length else min_length
            for i in range(min_length):
                new_name = name
                new_config = copy.deepcopy(old_config)
                for ikey in interlocking_key:
                    for ckey in config.keys():
                        if ikey in config[ckey].keys():
                            new_config[ckey][ikey] = gs_dict[key][ikey][i]
                            abbrev_list = ikey.split("_")
                            for abbrev in abbrev_list:
                                new_name += abbrev[0]
                            new_name += str(gs_dict[key][ikey][i])
                            break
                if depth+1 < gs_length:
                    generate_queue_config(new_config, new_name, depth+1)
                else:
                    config_list.append(new_config)
                    name_list.append(new_name)
                    #main(config, new_name)
        else:
            raise ValueError("elements must be a list type object or a dict type object")
            
    
    if parallel_strategy_on:
        if direct_parse_strategy_on:
            generate_queue_flatten_config(flatten_config, dir_name, 0)
            total_parse_list = []
            for config_element, name_element in zip(config_list, name_list):
                total_parse_list.append(["python", "parse_svae.py"] + flatten_config_to_parse(config_element) + ["--name", name_element])
            print(total_parse_list)
#             generate_queue_parse(initial_parse, dir_name, 0, 0)
#             for parse_element, name_element in zip(parse_list, name_list):
#                 total_parse_list.append(["python", "parse_svae.py"] + parse_element + ["--name", name_element])
        else:
            generate_queue_config(config, dir_name, 0)
    else:
        if direct_parse_strategy_on:
            generate_queue_parse(initial_parse, dir_name, 0, 0)
        else:
            generate_queue_flatten_config(flatten_config, dir_name, 0)
    
    if parallel_strategy_on:
        if direct_parse_strategy_on:
            for i in range((len(name_list)-1)//max_parallel_queues+1):
                p = mp.Pool(max_parallel_queues)
                p.map(subprocess.run, total_parse_list[max_parallel_queues*i:max_parallel_queues*(i+1)])
                p.close()
                if "gpu" in gs_key:
                    gpu_ids = gs_dict["gpu"]
                    memory_used = [int(get_gpu_info()[gpu_id]["memory.used"]) for gpu_id in gpu_ids]
                    while max(memory_used) > minimum_memory:
                        print("waiting in {}-th parallel computation".format(i+1))
                        time.sleep(10)
                        memory_used = [int(get_gpu_info()[gpu_id]["memory.used"]) for gpu_id in gpu_ids]
                else:
                    memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
                    while memory_used > minimum_memory:
                        print("waiting in {}-th parallel computation".format(i+1))
                        time.sleep(10)
                        memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
        else:
            for i in range((len(name_list)-1)//max_parallel_queues+1):
                p = mp.Pool(min(mp.cpu_count(), max_parallel_queues))
                p.starmap(main, zip(config_list[max_parallel_queues*i:max_parallel_queues*(i+1)], name_list[max_parallel_queues*i:max_parallel_queues*(i+1)]))
                p.close()
                memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
                while memory_used > minimum_memory:
                    print("waiting")
                    time.sleep(10)
                    memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
                #subprocess.run(["nvidia-smi"])
    else:
        if direct_parse_strategy_on:
            for parse_element, name_element in zip(parse_list, name_list):
                #print(name_element, parse_element)
                subprocess.run(["python", "parse_svae.py"] + parse_element + ["--name", name_element])
        else:
            for config_element, name_element in zip(config_list, name_list):
                parse_element = flatten_config_to_parse(config_element)
                #print(name_element, parse_element)
                subprocess.run(["python", "parse_svae.py"] + parse_element + ["--name", name_element])
    
            
            
if __name__ == "__main__":
    gs_main()