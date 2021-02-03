# PyTorch Implementation for Ensemble Kalman Variational Objective (EnKO)
This repository includes PyTorch implementation for "Ensemble Kalman Variational Objective: Nonlinear Latent Time-Series Model Inference by a Hybrid of Variational Inference and Ensemble Kalman Filter," an under review paper for ICML.
Following double-blind review policy of ICML, we hide our names and affiliations.
If our paper is accepted, we would like to open the identical repository in our mainly used GitHub page.

## Contents
This repository can implement following systems, models and data.
- Systems (SVAE/model/system/)
    - EnKO (ours): combination between ensemble Kalman filter (EnKF) and variational inference (VI)
    - [(Sequential) IWAE](https://arxiv.org/abs/1509.00519): sequential version of importance weighted auto-encoder
    - [FIVO](https://papers.nips.cc/paper/2017/hash/fa84632d742f2729dc32ce8cb5d49733-Abstract.html), also called [AESMC](https://openreview.net/forum?id=BJ8c3f-0b) or [VSMC](http://proceedings.mlr.press/v84/naesseth18a.html): combination between sequential Monte Carlo (SMC) and VI
    - [PSVO](https://www.semanticscholar.org/paper/Variational-Objectives-for-Markovian-Dynamics-with-Moretti-Wang/ccd5761c40305c4ffcc3a7cbc387ba1273895114): combination between forward filtering backward simulation (FFBSi) and VI
- Models (SVAE/model/)
    - [SVO](https://openreview.net/forum?id=HJg24U8tuE)
    - [AESMC](https://openreview.net/forum?id=BJ8c3f-0b)
    - [VRNN](https://papers.nips.cc/paper/2015/hash/b618c3210e934362ac261db280128c22-Abstract.html)
    - [SRNN](https://papers.nips.cc/paper/2016/hash/208e43f0e45c4c78cafadb83d2888cb6-Abstract.html)
    - [PSVO](https://www.semanticscholar.org/paper/Variational-Objectives-for-Markovian-Dynamics-with-Moretti-Wang/ccd5761c40305c4ffcc3a7cbc387ba1273895114)
        - Scripts of SVO, AESMC, and PSVO were started from [PSVO TensorFlow repository](https://github.com/amoretti86/PSVO) and we heavily refactored them for PyTorch implementation and new features.
- Data (data/)
    - FitzHugh-Nagumo Model
        - This generating process can be shown in "data_FHN.ipynb".
    - Lorenz Model
        - This generating process can be shown in "data_Lorenz.ipynb".
    - Allen Brain Atlas Dataset
        - This original data "SmallRipickledAllenDatawl" can be obtained in PSVO/data/allen directory in [PSVO repository](https://github.com/amoretti86/PSVO/tree/master/data/allen).
        - We obtained this original data and converted normalized data by preprocessing same as [PSVO](https://www.semanticscholar.org/paper/Variational-Objectives-for-Markovian-Dynamics-with-Moretti-Wang/ccd5761c40305c4ffcc3a7cbc387ba1273895114).

## How to Implement
How to implement the ensemble systems with the networks are follows
1. Set "config.json" file in SVAE directory. The details are described in later.
1. Run "run_svae.py" file by "python run_svae.py". Parallel implementaion of several conditions are carried out by "python gs_svae.py".

Experiments for variance of the gradient estimates described in Appendix A of our supplementary material are replicated by "SVAE/bias.ipynb"

## Requirement
We implemented our script in following environments.
- Cuda 10.2
- Python 3.8.5
- NumPy (numpy) 1.19.1
- matplotlib 3.3.1
- PyTorch (torch) 1.6.0
- torchvision 0.7.0

## Examples of Estimated Results
Latent trajectory inference for FHN data (left: true trajectory, right: inferred latent trajectory)
![fhn](https://github.com/ZoneMS/EnKO/tree/main/figs/quiver_plot2000.pdf)

Latent trajectory inference for Lorenz data (left: true trajectory, right: inferred latent trajectory)
![lorenz](https://github.com/ZoneMS/EnKO/tree/main/figs/traj_plot1540.pdf))

Trajectory inference for Allen Brain Atlas dataset (numbers corresponds to 10 test data)
![allen](https://github.com/ZoneMS/EnKO/tree/main/figs/traj_plot2000.pdf)


## Configuration
Configuration is divided into multiple blocks.
To reproduce results in our paper, we describe the detailed condition by parentheses ( ).
- train
    - batch_size (FHN=20, Lorenz=6, Allen=5): batch size for minibatch SGD.
    - lr (1e-3): learning rate for SGD.
    - epoch (2000): number of epochs for SGD.
    - conti_learn (false): whether continue from where the last learning ended. If true, load model and optimizer from the last learning whose epoch is *load_epoch*.
    - load_epoch (0): epoch for loading. if *conti_learn*, a user should set this value else 0.
    - train_rate (FHN=0.5, Lorenz=0.66): ratio for train data.
    - valid_rate (FHN=0.1, Lorenz=0.17): ratio for validation data. Residual ratio corresponds to test data.
    - num_workers (0): how many subprocesses to use for data loading in [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class. 0 means that the data will be loaded in main process.
    - seed (0~2): seed for randomness.
    - gpu (0~2): GPU ID.
- data
    - system: ensemble system for introducing a model. A user can choose from "EnKO", "FIVO", or "IWAE".
    - model: model for training. A user can choose from "SVO", "VRNN", "SRNN", "AESMC", or "PSVO". If the user set "PSVO", *system* must be "FIVO".
    - data_name: name of data. A user can choose from "FHN", "Lorenz", or "Allen".
    - scaling (FHN=abs-div, Lorenz=abs-div, Allen=null): scaling method for preprocessing data. A user can choose from "abs-div" (absolute division along each dimension), "min-max" (\[0,1\] scaling along each dimension), "th-abs-div" (similar to "abs-div", but a user can set minimum value for scaling by *scaling_factor*), "standard" (normalization along each dimension), or "null".
    - scaling_factor (0): scaling factor for "th-abs-div" scaling.
- training
    - scheduler (Plateau): scheduler for optimizing learning rate. A user can choose from "Plateau" (decay learning rate by *decay_rate* when a validation loss has stopped improving for *patience* epochs), "StepLR" (decays the learning rate of each parameter group by *decay_rate* every *decay_steps* epochs), or "null".
    - decay_steps (200): period of learning rate decay if *scheduler* is "StepLR".
    - decay_rate (0.7): multiplicative factor for learning rate decay.
    - patience (FHN=30, Lorenz=50, Allen=50): number of epochs with no improvement after which learning rate will be reduced if *scheduler* is "Plateau".
    - min_lr (3e-4): a lower bound on the learning rate of all parameter groups or each group respectively.
    - early_stop_patience (1000): number of epoch with no improvement after which early stopping is triggered.
    - clip (10): maximum value of norm of the gradient for [gradient norm clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).
    - pred_steps (20): number of predictive steps for calculating MSE and R2.
- network
    - Dz (FHN=2, Lorenz=3, Allen=3): dimension of latent variables.
    - Dh (32): dimension of hidden variables.
    - rnn (GRU): recurrent neural network in model. A user can choose from "RNN", "GRU", or "LSTM".
    - n_rnn_units (32): dimension of hidden variables for RNN.
    - n_particles (16): number of particles for ensemble system.
    - n_bw_particles (16): number of backward particles for PSVO.
    - dropout_ratio (0): dropout ratio.
    - n_layers (1): number of layers for RNN.
    - bias (false): bias for network.
    - output_dist (Gauss): output distribution for observation. A user can choose from "Gauss", "Laplace", or "Bernoulli".
    - ouput_fn (linear): activation function for final layer. A user can choose from "linear", "relu", "softmax", "softplus", "sigmoid", or "tanh".
    - residual_on (true): whethre use residual variables for "SRNN".
    - init_inference (true): whether compute initial inference of hidden variables for "VRNN". Although the original VRNN uses no initial inference, the inference should improve the performance.
    - sigma_init (1.0): initial variance of transition, inference, and emission for "SVO", "AESMC", and "PSVO".
    - sigma_min (0.1): minimum variance of transition, inference, and emission for "SVO", "AESMC", and "PSVO".
    - enc_steps (null): number of time-steps for initial inference for "VRNN", "SVO", and "PSVO". Null means use for all time-steps.
- enko (only valid for EnKO implementation)
    - filtering_method (inverse): filtering algoritm of the EnKF. A user can choose from "inverse" (default algorithm as described in our paper), "inverse-ide" (directly use diagonal variance of emission), "diag-inverse" (approximate sample covarince matrix by diagonal covariance), or "etkf-diag" (ensemble transform Kalman filter version).
    - inflation_method (RTPP): inflation method for the EnKF. A user can choose from "RTPP" ([relaxation to prior perturbation](https://journals.ametsoc.org/view/journals/mwre/132/5/1520-0493_2004_132_1238_ioieao_2.0.co_2.xml)), "RTPS" ([relaxation to prior spread](https://journals.ametsoc.org/view/journals/mwre/140/9/mwr-d-11-00276.1.xml)), or "null".
    - inflatio_factor (0.1): inflation factor for the inlfation method.
- print
    - print_freq: period of printing training process.
    - save_freq: period of saving training results.
    