"""
Spyder Editor
  
@author: fangshuyang (yangfs@hit.edu.cn)

"""

#### Define the parameters ####


import numpy as np
####################################################
####   MAIN PARAMETERS FOR FORWARD MODELING         ####
####################################################

peak_freq = 5.              # central frequency
peak_source_time = 1 / peak_freq  # the time (in secs) of the peak amplitude
dx        = 30.0               # step interval along x/z direction
dt        = 0.003             # time interval (e.g., 3ms)
num_dims  = 2             # dimension of velocity model
nz        = 100               # model shape of z dimension (depth)
ny        = 310               # model shape of y dimension (when num_dims>=2)
vmodel_dim = np.array([nz, ny])

total_t   = 6.                      # totle sampling time (unit: s)
nt        = int(total_t / dt)    # number of time sampling points
num_shots = 30                # nunmber of shots
num_sources_per_shot = 1  # number of sources per shot (note that you can have more than one source per shot, although in active seismic experiments,  it is normally only one)
num_receivers_per_shot = 310  # number of receivers per shot
source_spacing = np.floor(dx * ny / (num_shots+1))                         # space intervel between neigbouring sources
receiver_spacing = np.floor(dx * ny / (num_receivers_per_shot +1))   # space intervel between neigbouring receivers
source_depth = 0    # the index depth of sources, default is 0
receiver_depth = 0  # the index depths of receivers, default is 0
survey_pad = None
fix_value_depth = 0  # fix the inital value that is same as those in true model (won't be updated)

gfsigma    = 'line' # different type to get the initial  model
# if gfsigma = line, it means to generate the linear increased model, if gfsigma = const, it means to generate the constant model
if gfsigma == 'line':
    lipar = 1.0
elif gfsigma == 'lineminmax':
    lipar = 1.1
else:
    lipar = None
    
order          = 8     # precision order of finite difference in space domain
pml_width  = [0,10,10,10,0,0]   # pml padding width for boundary (0 for free surface and 10 for absorbing boundary)

AddNoise   = False # whether add AWGN to amplitude
if AddNoise == True:  
    noise_type = 'Gaussian'  
    if noise_type == 'Gaussian':
        noise_var = 10

#### parameters for bandpass filter of source wavelet #####
use_filter = False           # use filter if it's True
filter_type = 'highpass'  # type of filter
freqmin = 3                  # minimum frequency value for bandpass
freqmax = None           # maximum frequency value for bandpass
corners = 6                   # corners for filter
df = 1/dt

learnAWGN = False      # learn the noise level if it's True


####################################################
######       PARAMETERS FOR TRADITIONAL FWI   #######
####################################################
fwi_lr                     = 2     # learning rate for updating the model
fwi_batch               = 30   # number of batches for FWI (note this is different from batch in DL, please see the defination of Deepwave)
fwi_num_epochs    = 300  # number of iteration for passing entire data 
fwi_weight_decay   = 0     # weight decay ratio 

if fwi_weight_decay > 0:
    fwi_stepsize       = 100
    
AddTV                   = False  # applying ATV regularizer if True
if AddTV:
    alpha_tv             = 1.*1e-4  # weight for ATV 
    
fscale                     = 1e2       # scaler that multiplied to the normalized data for better back-propagating (this can also be implemented by changing learng rate)
data_norm             = False     # do data normalizationg if True
fwi_loss_type          = 'W1'      # loss type for FWI
savepoch               = 50         # number of epochs for saving the intermidate results
plot_ite                  = 300       # number of iteration for plotting


####################################################
####    PARAMETERS FOR FWIGAN     ###
####################################################
gan_v_lr                   = 3*1e1         # learning rate for updating the velocity model
gan_num_batches    = 6               # split data into (num_batches) batches for speed and reduced memory use
start_epoch              = 0 
gan_num_epochs     = 300           # pass through the entire dataset gan_num_epoches times totally
DFilter                     = 32             # number of channels of first conv block in discriminator
leak_value                = 0              # negative slope of the rectifier
gan_weight_decay   = 0               # weight decay ratio
if gan_weight_decay > 0:
    gan_stepsize        = 100            # step size for decreasing the learning rate
    
gan_d_lr                   = 1*1e-3       # learning rate for discriminator

criticIter                   = 6                # note that criticIter should be devided by num_batches
lamb                       = 10               # gradient penalty lambda parameter

####################################################
####    PARAMETERS FOR FWIDIP     ###
####################################################

"""partical parameters for pretaining"""
use_dip                  = True             # applying DIP if True
INPUT                    = 'noise'          # the input type of generative network
dip_pretrain_lr        = 0.01              # learning rate of pretraining
pretrain_num_iter   = 1000             # number of iteration for pretraining
dip_pretrain_loss    = 'L1'                # loss type for pretraining
OPT_OVER             = 'net'              # parameters for updating
OPTIMIZER            = 'adam'           # optimizer
k_channels             = 128               # number of channels of input
upsample_times     = 5                   # upsample times of network
output_channels    = 1                   # number of output channel
pad                       = 'zero'             # padding for upsampling 

reg_noise_std         = 0                  # add additional noise to input if it's larger than 0
LR_decrease          = False             # decrease the learning rate if True
if LR_decrease:
    OptiScStepSize  = 100               # step size size for decreasing the learning rate
    OptiScGamma   = 0.5                # weight decay ratio
else:
    OptiScStepSize  = None
    OptiScGamma   = None
    


"""partical parameters for FWIDIP"""

dip_v_lr                     = 1*1e-3      # learning rate for updating the  network in FWIDIP
dip_num_epochs      = 300            # number of total iterations
dip_num_batches     = 30              # number of batches
dip_weight_decay     = 0                # weight decay ratio
if dip_weight_decay > 0:
    dip_stepsize         = 100
dip_data_norm         = False          # do data normalizationg if True