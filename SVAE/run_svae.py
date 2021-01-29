import os, sys, json
import time, shutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import load_data, transform_data, get_dataset, get_model, plot_loss, plot_predictive_result
from test_svae import test_main

def main(config, name, number_of_data=""):
    date = "210129"
    system = config["data"]["system"] # FIVO, EnKO, IWAE
    model_name = config["data"]["model"] # VRNN, AESMC, SRNN, SVO, PSVO (only FIVO)
    data_name = config["data"]["data_name"] # Lorenz, FHN, Allen
    number_of_date = ""
    result_dir = "../results/{}_{}{}".format(date,
                                            name,
                                            number_of_date)
    print(result_dir)
    
    ## Set seed and cuda
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
    obs_train, obs_valid, _, Dx = load_data(config)
    obs_train, obs_valid, svec = transform_data(obs_train, obs_valid, config)
    train_loader, valid_loader = get_dataset(config, obs_train, obs_valid)
    
    conti_learn = config["train"]["conti_learn"]
    load_epoch = config["train"]["load_epoch"]
    num_epochs = config["train"]["epoch"]
    total_epochs = load_epoch + num_epochs
    init_lr = config["train"]["lr"]
    
    scheduler_name = config["training"]["scheduler"]
    decay_rate = config["training"]["decay_rate"]
    decay_steps = config["training"]["decay_steps"]
    patience = config["training"]["patience"]
    early_stop_patience = config["training"]["early_stop_patience"]
    clip = config["training"]["clip"]
    min_lr = config["training"]["min_lr"]
    pred_steps = config["training"]["pred_steps"]
    
    display_steps = config["print"]["print_freq"]
    save_steps = config["print"]["save_freq"]
    
    model, loss_name_list = get_model(config, Dx, device)
    n_losses = len(loss_name_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
 
    if not os.path.exists(result_dir.rsplit("/",1)[0]):
        os.mkdir(result_dir.rsplit("/",1)[0])
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    elif conti_learn:
        with open(os.path.join(result_dir, "config{}.json".format(load_epoch)), "w") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        #shutil.copy("config.json", os.path.join(result_dir, "config{}.json".format(load_epoch)))
        state_dict = torch.load(os.path.join(result_dir, "state_dict_{}.pth".format(load_epoch)))
        model.load_state_dict(state_dict)
        optimizer_state_dict = torch.load(os.path.join(result_dir, "optimizer_state_dict_{}.pth".format(load_epoch)))
        optimizer.load_state_dict(optimizer_state_dict)
    
    if not conti_learn:
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        #shutil.copy("config.json", os.path.join(result_dir, "config.json"))
    
    
    if scheduler_name=="StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif scheduler_name=="Plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_rate, patience=patience, min_lr=min_lr)
    
    
    # Define functions
    def train(epoch):
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
            if model_name=="SRNN":
                _loss, _  = model(data, covariate)
            else:
                _loss, _  = model(data)
            times[0] += time.time() - start_time
            start_time = time.time()
            _loss[0].backward()
            times[1] += time.time() - start_time
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            
            for i in range(n_losses):
                loss[i] += _loss[i].item()

        loss /= len(train_loader)
                
        if epoch%display_steps==0:
            print_contents = "Train Epoch: [{}/{}]".format(epoch, num_epochs)
            for i, loss_name in enumerate(loss_name_list):
                print_contents += "\t {}: {:.6f}".format(
                                    loss_name,
                                    loss[i])
            print(print_contents)
        return loss, times
    
    
    def valid(epoch):
        """uses test data to evaluate likelihood of the model"""
        start_time = time.time()
        model.eval()
        loss = np.zeros(n_losses)
        pred_evals = np.zeros((4, pred_steps))

        for (data, covariate) in valid_loader:                                            
            data = Variable(data)
            data = data.transpose(0,1).to(device)
            covariate = Variable(covariate) # (bs,T,Dx)/(bs,T)
            covariate = covariate.transpose(0,1).to(device) # (T,bs,Dx)/(T,bs)

            if model_name=="SRNN":
                _loss, (Z, _, H)  = model(data, covariate)
            elif model_name in ["SVO", "PSVO", "AESMC"]:
                _loss, (Z, _, _)  = model(data)
            elif model_name in ["VRNN"]:
                _loss, (_, _, Z)  = model(data)
            else:
                _loss, _  = model(data)
            
            for i in range(n_losses):
                loss[i] += _loss[i].item()
            
            _TV = None
            if model_name=="SRNN":
                _MSE, _TV = model.calculate_mse(data, Z, H, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
            else:
                _MSE, _TV = model.calculate_mse(data, Z, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
            _MSE = _MSE.data.cpu().numpy()
            MSE = _MSE.sum(axis=2) #(ps,bs)
            sMSE = (_MSE*svec*svec).sum(axis=2) #(ps,bs)
            if _TV is None:
                TV = sTV = 1
            else:
                _TV = _TV.data.cpu().numpy()
                TV = _TV.sum(axis=2) #(ps,bs)
                sTV = (_TV*svec*svec).sum(axis=2) #(ps,bs)
            pred_evals[0] += MSE.sum(axis=1)
            pred_evals[1] += sMSE.sum(axis=1)
            pred_evals[2] += (1 - MSE/TV).sum(axis=1) #R2
            pred_evals[3] += (1 - sMSE/sTV).sum(axis=1) #scaling R2

        loss /= len(valid_loader)
        pred_evals /= len(obs_valid)
        
        if epoch%display_steps==0:
            print_contents = "====> Test set loss:"
            for i, loss_name in enumerate(loss_name_list):
                print_contents += " {} = {:.4f}".format(loss_name,
                                                        loss[i])
            print(print_contents)
        return loss, pred_evals, time.time() - start_time
    
    
    ## Train model
    def execute():
        train_loss = np.zeros([total_epochs, n_losses])
        valid_loss = np.zeros([total_epochs, n_losses])
        pred_evals = np.zeros([total_epochs, 4, pred_steps]) #0:MSE,1:sMSE,2:R2,3:sR2
        time_record = np.zeros([total_epochs, 4]) #0:total,1:fw,2:bw,3:valid
        min_valid_loss = 10e+8
        early_stop_count = 0
        if conti_learn:
            train_loss[:load_epoch] = np.load(os.path.join(result_dir, "train_loss.npy"))[:load_epoch]
            valid_loss[:load_epoch] = np.load(os.path.join(result_dir, "valid_loss.npy"))[:load_epoch]
            pred_evals[:load_epoch] = np.load(os.path.join(result_dir, "pred_evals.npy"))[:load_epoch]
            time_record[:load_epoch] = np.load(os.path.join(result_dir, "time_record.npy"))[:load_epoch]
        start_time = time.time()

        for epoch in range(load_epoch+1, total_epochs+1):
            epoch_start_time = time.time()
            
            # training + testing
            train_loss[epoch-1], time_record[epoch-1,1:3] = train(epoch)
            valid_loss[epoch-1], pred_evals[epoch-1], time_record[epoch-1,3] = valid(epoch)
            
            if scheduler_name=="StepLR":
                scheduler.step()
            elif scheduler_name=="Plateau":
                scheduler.step(valid_loss[epoch-1,0])

            # duration
            duration = int(time.time() - start_time)
            second = int(duration%60)
            remain = int(duration//60)
            minute = int(remain%60)
            hour = int(remain//60)
            print("Duration: {} hour, {} min, {} sec.".format(hour, minute, second))
            remain = (total_epochs - epoch) * duration / (epoch - load_epoch)
            second = int(remain%60)
            remain = int(remain//60)
            minute = int(remain%60)
            hour = int(remain//60)
            print("Estimated Remain Time: {} hour, {} min, {} sec.".format(hour, 
                                                                           minute, 
                                                                           second))
            time_record[epoch-1,0] = time.time() - epoch_start_time

            # saving model
            if epoch % save_steps == 0:
                torch.save(model.state_dict(), os.path.join(result_dir, 
                                        'state_dict_'+str(epoch)+'.pth'))
                torch.save(optimizer.state_dict(), os.path.join(result_dir,
                                            'optimizer_state_dict_'+str(epoch)+'.pth'))
                print('Saved model to state_dict_'+str(epoch)+'.pth')
                np.save(os.path.join(result_dir, "time_record.npy"), time_record[:epoch])
                np.save(os.path.join(result_dir, "train_loss.npy"), train_loss[:epoch])
                np.save(os.path.join(result_dir, "valid_loss.npy"), valid_loss[:epoch])
                np.save(os.path.join(result_dir, "pred_evals.npy"), pred_evals[:epoch])
                
                plot_loss(epoch, train_loss, valid_loss, loss_name_list, result_dir)
                plot_predictive_result(epoch, pred_evals[epoch-1], pred_steps, result_dir, ["MSE", "sMSE", "R2", "sR2"])
                
            ## early stopping
            if valid_loss[epoch-1,0] < min_valid_loss:
                min_valid_loss = valid_loss[epoch-1,0]
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count >= early_stop_patience:
                print("Early Stopping is triggered at epoch={}".format(epoch))
                break
        
        test_main(result_dir, save_steps*np.argmin(valid_loss[::save_steps,0])+1)
        
    
    
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
    model_name = config["data"]["model"] # VRNN, MINN, AESMC, STORN, SRNN, SVO, SVO-II, MINN-SVO, PSVO (only FIVO)
    data_name = config["data"]["data_name"] # Lorenz, FHN, TEPCO, Allen, Mocap
    name = "{}{}_{}".format(system_name, model_name, data_name)
    main(config, name)