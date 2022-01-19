# PyTorch Implementation for Ensemble Kalman Variational Objective (EnKO)
This repository includes PyTorch implementation for Ensemble Kalman Variational Objective: A Variational Inference Framework for Sequential Variational Auto-Encoders," an under review paper for JMLR.


## Contents
This repository can implement following systems, models and data.
- Systems (SVAE/model/system/)
    - EnKO (ours): combination between ensemble Kalman filter (EnKF) and variational inference (VI)
    - [(Sequential) IWAE](https://arxiv.org/abs/1509.00519): sequential version of importance weighted auto-encoder
    - [FIVO](https://papers.nips.cc/paper/2017/hash/fa84632d742f2729dc32ce8cb5d49733-Abstract.html), also called [AESMC](https://openreview.net/forum?id=BJ8c3f-0b) or [VSMC](http://proceedings.mlr.press/v84/naesseth18a.html): combination between sequential Monte Carlo (SMC) and VI
- Models (SVAE/model/)
    - [SVO](https://openreview.net/forum?id=HJg24U8tuE)
    - [AESMC](https://openreview.net/forum?id=BJ8c3f-0b)
    - [VRNN](https://papers.nips.cc/paper/2015/hash/b618c3210e934362ac261db280128c22-Abstract.html)
    - [SRNN](https://papers.nips.cc/paper/2016/hash/208e43f0e45c4c78cafadb83d2888cb6-Abstract.html)
- Data (data/)
    - FitzHugh-Nagumo Model
        - This generating process can be shown in "data_FHN.ipynb".
    - Lorenz Model
        - This generating process can be shown in "data_Lorenz.ipynb".
    - Walking Dataset from [CMU Motion Capture Library](http://mocap.cs.cmu.edu/)
    - Rotating MNIST Dataset from [this repository](https://github.com/ChaitanyaBaweja/RotNIST)

## How to Implement
How to implement the ensemble systems with the networks are follows
1. Set "config.json" file in SVAE directory. The details are described in later.
1. Run "run_svae.py" file by "python run_svae.py". Parallel implementaion of several conditions are carried out by "python gs_svae.py".

Experiments for variance of the gradient estimates described in Appendix A of our supplementary material are replicated in "SVAE/bias.ipynb"

## Requirement
We implemented our script in following environments.
- Cuda 10.2
- Python 3.8.5
- NumPy (numpy) 1.19.1
- matplotlib 3.3.1
- PyTorch (torch) 1.6.0
- torchvision 0.7.0
- comet-ml 3.12.0

## Examples of Estimated Results
True and inferred latent dynamics and trajectories for FHN data (left: true dynamics, right: inferred latent dynamics).
![true dynamics](figs/test_quiver_plot_orig.pdf)
![inferred dynamics](figs/test_quiver_plot_recon.pdf)

True and inferred latent trajectories for Lorenz data (left: true trajectories, right: inferred latent trajectories).
![true trajectories](figs/test_traj_plot_ax_orig.pdf)
![inferred trajectories](figs/test_traj_plot_ax_recon.pdf)

Long prediction results for the walking dataset.
We inferred the initial latent state and predicted the values of the observations at all remaining time points according to the learned generative model.
The black times represent the observed points, the solid blue line represents the predicted mean, and the dark and light blue widths represent the predicted mean plus or minus standard deviation and two standard deviations, respectively.
The text in the figure shows the variable names.
The vz, vx, and vy correspond to velocities, alpha, beta, and gamma correspond to Euler angles, and l and r correspond to left and right.
![walking](figs/mocap_rtps_init_running.pdf)

True images and prediction results for rotating MNIST dataset.
![rmnist](figs/rmnist_plotnew.pdf)


## Configuration
Configuration is divided into multiple blocks.
To reproduce results in our paper, we describe the detailed conditions by parentheses (F: Fitz-Hugh Nagumo, L: Lorenz, W: Walking, R: RMNIST).
- train
    - batch_size (F:20, L:6, W:4, R:40): batch size for minibatch SGD.
    - lr (1e-3): learning rate for SGD.
    - epoch (FLW:2000, R:3000): number of epochs for SGD.
    - conti_learn (false): whether continue from where the last learning ended. If true, load model and optimizer from the last learning whose epoch is *load_epoch*.
    - load_epoch (0): epoch for loading. if *conti_learn*, a user should set this value else 0.
    - train_rate (F:0.5, L:0.66): ratio for train data.
    - valid_rate (F:0.1, L:0.17): ratio for validation data. Residual ratio corresponds to test data.
    - train_num (W:16, R:360): number of train samples.
    - valid_num (W:3, R:40): number of valid samples.
    - num_workers (0): how many subprocesses to use for data loading in [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class. 0 means that the data will be loaded in main process.
    - seed (1~3): seed for randomness.
    - gpu (0): GPU ID.
- data
    - system: ensemble system for introducing a model. A user can choose from "EnKO", "FIVO", or "IWAE".
    - outer_model (FLW:null, R:StyleConv): outer VAE structure. A use can choose from null, "Conv", and "StyleConv".
    - model (SVO): model for training. A user can choose from "SVO", "VRNN", and "AESMC".
    - data_name: name of data. A user can choose from "FHN", "Lorenz", "Mocap" or "rmnist".
    - scaling (FL:abs-div, WR:null): scaling method for preprocessing data. A user can choose from "abs-div" (absolute division along each dimension), "min-max" (\[0,1\] scaling along each dimension), "th-abs-div" (similar to "abs-div", but a user can set minimum value for scaling by *scaling_factor*), "standard" (normalization along each dimension), or "null".
    - scaling_factor (0): scaling factor for "th-abs-div" scaling.
- training
    - scheduler (Plateau): scheduler for optimizing learning rate. A user can choose from "Plateau" (decay learning rate by *decay_rate* when a validation loss has stopped improving for *patience* epochs), "StepLR" (decays the learning rate of each parameter group by *decay_rate* every *decay_steps* epochs), or "null".
    - decay_steps (FL:200,R:20): period of learning rate decay for *scheduler* of "StepLR".
    - decay_rate (FLW:0.7,R:0.8): multiplicative factor for learning rate decay.
    - patience (FR:30, LW:50): number of epochs with no improvement after which learning rate will be reduced for *scheduler* of "Plateau".
    - clip (FLW:10,R:150): maximum value of norm of the gradient for [gradient norm clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).
    - min_lr (3e-4): a lower bound on the learning rate of all parameter groups or each group respectively.
    - early_stop_patience (3000): number of epoch with no improvement after which early stopping is triggered.
    - pred_steps (FLW:20, R:15): number of predictive steps for calculating MSE and R2.
- network
    - Dz (F:2, L:3, W:6, R:2): dimension of latent variables.
    - Dh (FL:32, W:100, R:50): dimension of hidden variables.
    - rnn (GRU): recurrent neural network in model. A user can choose from "RNN", "GRU", or "LSTM".
    - n_rnn_units (FL:32, W:100, R:50): dimension of hidden variables for RNN.
    - n_particles (FL:16, W:128, R:32): number of particles for ensemble system.
    - n_layers (1): number of layers for RNN.
    - bias (false): bias for network.
    - dropout_ratio (0): dropout ratio.
    - output_dist (Gauss): output distribution for SVAE. A user can choose from "Gauss", "Laplace", or "Bernoulli".
    - ouput_fn (linear): activation function for final layer of SVAE. A user can choose from "linear", "relu", "softmax", "softplus", "sigmoid", or "tanh".
    - outer_output_dist (R:Bernoulli): output distribution for outer VAE. A user can choose from "Gauss", "Laplace", or "Bernoulli".
    - ouput_fn (R:Sigmoid): activation function for final layer of outer VAE. A user can choose from "linear", "relu", "softmax", "softplus", "sigmoid", or "tanh".
    - init_inference (true): whether compute initial inference of hidden variables for "VRNN". Although the original VRNN uses no initial inference, the inference should improve the performance.
    - sigma_init (1.0): initial variance of transition, inference, and emission for "SVO" and "AESMC".
    - sigma_min (0.1): minimum variance of transition, inference, and emission for "SVO" and "AESMC".
    - enc_steps (null): number of time-steps for initial inference for "VRNN", "SVO", and "PSVO". Null means use for all time-steps.
    - loss_type (EnKO:sumprod, FIVO:prodsum, IWAE:sumprod): loss type for computation. EnKO and IWAE should be "sumprod", and FIVO should be "prodsum".
- enko (only valid for EnKO implementation)
    - filtering_method (inverse): filtering algoritm of the EnKF. A user can choose from "inverse" (default algorithm as described in our paper), "inverse-ide" (directly use diagonal variance of emission), "diag-inverse" (approximate sample covarince matrix by diagonal covariance), or "etkf-diag" (ensemble transform Kalman filter version).
    - inflation_method: inflation method for the EnKF. A user can choose from "RTPP" ([relaxation to prior perturbation](https://journals.ametsoc.org/view/journals/mwre/132/5/1520-0493_2004_132_1238_ioieao_2.0.co_2.xml)), "RTPS" ([relaxation to prior spread](https://journals.ametsoc.org/view/journals/mwre/140/9/mwr-d-11-00276.1.xml)), or "null".
    - inflatio_factor: inflation factor for the inlfation method. This value should be from 0 to 1.
- conv (only valid for outer Convolution VAE)
    - filter_enc (R:\[16,32,64\]): filter size of encoder.
    - kernel_enc (R:\[5,5,5\]): kernel size of encoder.
    - stride_enc (R:\[2,2,2\]): stride size of encoder.
    - padding_enc (R:\[2,2,2\]): padding size of encoder.
    - bn_enc (R:\[true,true,true\]): whether batch normlize in encoder.
    - filter_dec (R:\[64,32,16,8\]): filter size of decoder.
    - kernel_dec (R:\[3,5,5,5\]): kernel size of decoder.
    - stride_dec (R:\[1,2,2,1\]): stride size of decoder.
    - padding_dec (R:\[0,1,1,2\]): padding size of decoder.
    - output_padding_dec (R:\[0,0,1,0\]): output padding size of decoder.
    - bn_dec (R:\[true,true,true\]): whether batch normlize in decoder.
    - conv_activation_fn: activation function for outer VAE. A user can choose from "linear", "relu", "softmax", "softplus", "sigmoid", or "tanh".
- print
    - print_freq: period of printing training process.
    - save_freq: period of saving training results.
    