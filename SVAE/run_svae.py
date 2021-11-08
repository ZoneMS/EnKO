import os, sys, json
import time, shutil, itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import comet_ml
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import load_data, transform_data, get_dataset, get_model, plot_loss, plot_predictive_result
from test_svae import test_main


def main(config, name):
    start_time = time.time()
    experiment_key = os.environ.get("COMET_experiment_key", config["train"]["conti_key"])
    continue_run = False
    if (experiment_key is not None):
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API() # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_id(experiment_key)
        except Exception:
            api_experiment = None
        if api_experiment is not None:
            continue_run = True
            # We can get the last details logged here, if logged:
            #print(api_experiment.get_parameters_summary())
            step = int(api_experiment.get_parameters_summary("train_steps")["valueCurrent"])
            epoch = int(api_experiment.get_parameters_summary("train_epochs")["valueCurrent"])
    
    if continue_run:
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=experiment_key,
            log_env_details=True, # to continue env logging
            log_env_gpu=True,     # to continue GPU logging
            log_env_cpu=True,     # to continue CPU logging
        )
        # Retrieved from above APIExperiment
        experiment.set_step(step)
        experiment.set_epoch(epoch)
        
        load_epoch = epoch
        num_epochs = config["train"]["epoch"]
        
        result_dir = api_experiment.get_parameters_summary("result_dir")["valueCurrent"]
        f = open(os.path.join(result_dir, "config.json"), "r")
        config = json.load(f)
        f.close()
        print("load config from {}/config.json".format(result_dir))
    else:
        data_name = config["data"]["data_name"] # Lorenz, FHN, TEPCO, Allen, Mocap, bball, rmnist
        experiment = comet_ml.Experiment(project_name="SVAE_{}".format(data_name))
        load_epoch = 0
        num_epochs = config["train"]["epoch"]

        result_dir = os.path.join("../results", name)
        print("set result directory to {}".format(result_dir))
        
    system = config["data"]["system"] # FIVO, EnKO, IWAE
    outer_model = config["data"]["outer_model"] # Conv
    model_name = config["data"]["model"] # VRNN, MINN, AESMC, AESMCp, STORN, SRNN, SVO, SVOD, SVOp, SVOg, SVO-II, MINN-SVO, PSVO (only FIVO), NODE
    data_name = config["data"]["data_name"] # Lorenz, FHN, TEPCO, Allen, Mocap, bball, rmnist
    
    flatten_config = {}
    for key in config.keys():
        flatten_config.update(config[key])
    flatten_config["result_dir"] = result_dir
    if not continue_run:
        experiment.log_parameters(flatten_config)
    
    ## Set seed and cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("Set seed {}.".format(config["train"]["seed"]))
    
    cuda = torch.cuda.is_available()
    if cuda:
        print("cuda is available")
        device = torch.device('cuda', config["train"]["gpu"])
    else:
        print("cuda is not available")
        device = torch.device("cpu")
    torch.manual_seed(config["train"]["seed"])
    if cuda:
        torch.cuda.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    
#     cuda = False
#     device = torch.device('cpu')
#     np.random.seed(config["train"]["seed"])
#     torch.autograd.set_detect_anomaly(True)
    
    
    ## Read data and set parameters
    obs_train, obs_valid, _, time_train, time_valid, _, Dx = load_data(config)
    obs_train, obs_valid, svec = transform_data(obs_train, obs_valid, config)
    train_loader, valid_loader = get_dataset(config, obs_train, obs_valid, time_train, time_valid)
    
    total_epochs = load_epoch + num_epochs
    init_lr = config["train"]["lr"]
    
    scheduler_name = config["training"]["scheduler"]
    decay_rate = config["training"]["decay_rate"]
    decay_steps = config["training"]["decay_steps"]
    patience = config["training"]["patience"]
    early_stop_patience = config["training"]["early_stop_patience"]
    saturated_patience = config["training"]["saturated_patience"]
    clip = config["training"]["clip"]
    min_lr = config["training"]["min_lr"]
    pred_steps = config["training"]["pred_steps"]
    pretrain_list = config["training"]["pretrain_list"]
    pretrain_epochs = config["training"]["pretrain_epochs"]
    pretrain_epochs = pretrain_epochs if pretrain_epochs is not None else 0
    
    metrics_total_dict = {"MSE":["MSE", "sMSE", "R2", "sR2"], "FIP":["NIP", "FIP"]}
    if outer_model in ["Conv", "StyleConv"]:
        metrics_total_dict["MSE"] = ["MSE", "R2"]
    evaluation_metrics = config["test"]["evaluation_metrics"]
    total_evaluation_metrics = sum([metrics_total_dict[i] for i in evaluation_metrics], [])
    
    display_steps = config["print"]["print_freq"]
    save_steps = config["print"]["save_freq"]
    
    model, loss_name_list = get_model(config, Dx, device)
    n_losses = len(loss_name_list)
    for name, param in model.named_parameters():
        print(name, param.shape)
    
    if pretrain_list is not None and pretrain_epochs > 0:
        pretrain_keys, posttrain_keys = [], []
        pretrain_params, posttrain_params = [], []
        for name, param in model.named_parameters():
            if np.any([i in name for i in pretrain_list]):
                pretrain_keys.append(name)
                pretrain_params.append(param)
            else:
                posttrain_keys.append(name)
                posttrain_params.append(param)
        print("pretrain", pretrain_keys)
        print("posttrain", posttrain_keys)
        optimizer = torch.optim.Adam(pretrain_params, lr=init_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
 
    if not os.path.exists(result_dir.rsplit("/",1)[0]):
        os.mkdir(result_dir.rsplit("/",1)[0])
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        
    if continue_run:
        state_dict = torch.load(os.path.join(result_dir, "state_dict_new.pth"))
        model.load_state_dict(state_dict)
        optimizer_state_dict = torch.load(os.path.join(result_dir, "optimizer_state_dict_new.pth"))
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        #shutil.copy("config.json", os.path.join(result_dir, "config.json"))
    
    
    if scheduler_name=="StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif scheduler_name=="Plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_rate, patience=patience, min_lr=min_lr)
    print("training preparation is end: {} sec".format(time.time() - start_time))
    
    
    # Define functions
    def train(epoch, saturated_on):
        model.train()
        loss = np.zeros(n_losses)
        times = np.zeros(2)

        for batch_idx, (data, covariate) in enumerate(train_loader):
            # transform data
            data = Variable(data) # (bs,T,Dx)/(bs,T,c,h,w)
            data = data.transpose(0,1).to(device) # (T,bs,Dx)/(T,bs,c,h,w)
            covariate = Variable(covariate) # (bs,T)/(bs,T,Dx)/(bs,T,c,h,w)
            covariate = covariate.transpose(0,1).to(device) # (T,bs)/(T,bs,Dx)/(T,bs,c,h,w)

            # forward + backward + optimize
            optimizer.zero_grad()
            start_time = time.time()
            args = []
            if outer_model is not None:
                args += [epoch, True]
            else:
                args += [saturated_on]
                    
            _loss, _ = model(data, *args)
            times[0] += time.time() - start_time
            start_time = time.time()
            _loss[0].backward()
            times[1] += time.time() - start_time
            nn.utils.clip_grad_norm_(model.parameters(), clip)
#             if epoch in range(10,12):
#                 print("epoch={}".format(epoch))
#                 for name, param in model.named_parameters():
#                     print(name, param.grad)
            optimizer.step()
            experiment.log_parameter("steps", (epoch-1)*len(train_loader)+batch_idx+1)
            
            
            for i in range(n_losses):
                #print(loss_name_list[i], _loss[i].item())
                loss[i] += _loss[i].item()

        loss /= len(train_loader)
        
        for i, loss_name in enumerate(loss_name_list):
            experiment.log_metric("train {}".format(loss_name), loss[i], step=epoch)
                
        if epoch%display_steps==0:
            print_contents = "Train Epoch: [{}/{}]".format(epoch, total_epochs)
            for i, loss_name in enumerate(loss_name_list):
                print_contents += "\t {}: {:.6f}".format(
                                    loss_name,
                                    loss[i])
            print(print_contents)

    
    
    def valid(epoch, saturated_on):
        """uses test data to evaluate likelihood of the model"""
        start_time = time.time()
        model.eval()
        loss = np.zeros(n_losses)
        pred_evals = np.zeros((len(total_evaluation_metrics), pred_steps))
        
        for (data, covariate) in valid_loader:
            data = Variable(data)
            data = data.transpose(0,1).to(device)
            covariate = Variable(covariate) # (bs,T,Dx)/(bs,T)
            covariate = covariate.transpose(0,1).to(device) # (T,bs,Dx)/(T,bs)

            H = None
            if outer_model is None:
                if model_name in ["SVO", "AESMC"]:
                    _loss, (Z, _, _)  = model(data, saturated_on)
                elif model_name in ["VRNN"]:
                    if time_train is not None:
                        _loss, (_, _, Z)  = model(data, saturated_on, covariate)
                    else:
                        _loss, (_, _, Z)  = model(data, saturated_on)
                else:
                    _loss, _  = model(data)
            elif outer_model=="Conv":
                if model_name in ["SVO", "AESMC"]:
                    _loss, (Z, _, _, _)  = model(data, epoch, False, saturated_on)
                elif model_name in ["VRNN"]:
                    _loss, (_, _, Z, _)  = model(data, epoch, False)
                else:
                    _loss, _  = model(data, epoch, False)
            elif outer_model=="StyleConv":
                if model_name in ["SVO", "AESMC"]:
                    _loss, (Z, _, a_style, _, _)  = model(data, epoch, False, saturated_on)
                elif model_name in ["VRNN"]:
                    _loss, (_, _, a_style, Z, _)  = model(data, epoch, False)
            
            for i in range(n_losses):
                loss[i] += _loss[i].item()
            
            eval_count = 0
            
            if outer_model in ["Conv"]:
                _metrics = model.calculate_predictive_metrics(data, Z, H, pred_steps, evaluation_metrics) #(nm,ps,bs)
                pred_evals += _metrics.sum(2) #(nm,ps)
            elif outer_model == "StyleConv":
                _metrics = model.calculate_predictive_metrics(data, Z, a_style, H, pred_steps, evaluation_metrics) #(nm,ps,bs)
                pred_evals += _metrics.sum(2) #(nm,ps)
            else:
                if "MSE" in evaluation_metrics:
                    _TV = None
                    if time_train is None:
                        _MSE, _TV = model.calculate_mse(data, Z, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
                    else:
                        _MSE, _TV = model.calculate_mse(data, Z, pred_steps, covariate) #(ps,bs,Dx),(ps,bs,Dx)
                    
                    #_MSE = _MSE.data.cpu().numpy()
                    MSE = _MSE.sum(axis=2) #(ps,bs)
                    sMSE = (_MSE*svec*svec).sum(axis=2) #(ps,bs)
                    if _TV is None:
                        TV = sTV = 1
                    else:
                        #_TV = _TV.data.cpu().numpy()
                        TV = _TV.sum(axis=2) #(ps,bs)
                    sTV = (_TV*svec*svec).sum(axis=2) #(ps,bs)
                    pred_evals[0] += MSE.sum(axis=1)
                    pred_evals[1] += sMSE.sum(axis=1)
                    pred_evals[2] += (1 - MSE/TV).sum(axis=1) #R2
                    pred_evals[3] += (1 - sMSE/sTV).sum(axis=1) #scaling R2
                    eval_count += 4
                    
                
        loss /= len(valid_loader)
        pred_evals /= len(obs_valid)
        
        for i, loss_name in enumerate(loss_name_list):
            experiment.log_metric("valid {}".format(loss_name), loss[i], step=epoch)
        for i, eval_name in enumerate(total_evaluation_metrics):
            for j in range(pred_steps):
                experiment.log_metric("{}-step {}".format(j+1, eval_name), pred_evals[i,j], step=epoch)
        
        if epoch%display_steps==0:
            print_contents = "====> Test set loss:"
            for i, loss_name in enumerate(loss_name_list):
                print_contents += " {} = {:.4f}".format(loss_name,
                                                        loss[i])
            print(print_contents)
        return loss, pred_evals
    
    
    ## Train model
    def execute():
        min_valid_loss = 10e+8
        early_stop_count = 0
        saturated_on = False

        with experiment.train():
            for epoch in range(load_epoch+1, total_epochs+1):
                # add post-training parameters
                if epoch==pretrain_epochs and pretrain_list is not None:
                    for params in posttrain_params:
                        optimizer.add_param_group({"params":params})

                # training + testing
                experiment.log_parameter("epochs", epoch)
                train(epoch, saturated_on)
                valid_loss, pred_evals = valid(epoch, saturated_on)

                if scheduler_name=="StepLR":
                    scheduler.step()
                elif scheduler_name=="Plateau":
                    scheduler.step(valid_loss[0])


                # saving model
                if epoch % save_steps == 0:
                    torch.save(model.state_dict(), os.path.join(result_dir, 
                                            'state_dict_new.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(result_dir,
                                                'optimizer_state_dict_new.pth'))
                    print('Saved model to state_dict_new.pth')

                ## early stopping
                if valid_loss[0] < min_valid_loss:
                    min_valid_loss = valid_loss[0]
                    early_stop_count = 0
                    torch.save(model.state_dict(), os.path.join(result_dir, 'state_dict.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(result_dir, 'optimizer_state_dict.pth'))
                    print('Saved model to state_dict.pth')
                    plot_predictive_result(pred_evals, pred_steps, result_dir, total_evaluation_metrics)
                else:
                    early_stop_count += 1

                if early_stop_count >= saturated_patience:
                    saturated_on = True

                if early_stop_count >= early_stop_patience:
                    print("Early Stopping is triggered at epoch={}".format(epoch))
                    break
        
        with experiment.test():
            test_main(result_dir, experiment, valid_on=True)
            test_main(result_dir, experiment, valid_on=False)
        
    
    
    ## Excute
    # measure for culabas rutine error
    execute()


    
if __name__ == "__main__":
    f = open("config.json", "r")
    config = json.load(f)
    f.close()
    system = config["data"]["system"] # FIVO, EnKO, IWAE
    if system is None:
        system_name = ""
    else:
        system_name = "{}_".format(system)
    outer_model = config["data"]["outer_model"] # Conv
    model_name = config["data"]["model"] # VRNN, MINN, AESMC, STORN, SRNN, SVO, SVO-II, MINN-SVO, PSVO (only FIVO)
    if outer_model is not None:
        model_name = outer_model + "-" + model_name
    data_name = config["data"]["data_name"] # Lorenz, FHN, TEPCO, Allen, Mocap
    date = config["train"]["date"]
    number_of_date = config["train"]["number_of_date"]
    name = "{}_{}{}_{}{}".format(date, system_name, model_name, data_name, number_of_date)
    main(config, name)