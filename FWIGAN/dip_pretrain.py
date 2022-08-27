#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fangshuyang (yangfs@hit.edu.cn)

####################    Deep Image Prior Training   ########################

reference: 
   https://github.com/DmitryUlyanov/deep-image-prior

########################################################################
"""

###### Import Libaries, Parameters and Paths  ######
from LibConfig import *
from PathConfig import *
from ParamConfig import *

######  Check CUDNN and GPU  ######
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


# check wheter use dip or not
if not use_dip:
    assert False, "Please choose the correct process!!!"
    
### Load true velocity model  ###
model_true = loadtruemodel(data_path, num_dims, vmodel_dim).to(device)
print(' maximum value of true model: ', model_true.max())

### Load  initial model guess ###
initfile = ResultPath+str(dataname)+'_initmodel.mat'
print('Loading initial model...')
_, model_init = loadinitmodel(initfile,device) 
model_init = model_init.reshape(1,1,nz,ny).to(device)    

### Define the generative network #############

netG = skip(num_input_channels=k_channels, \
            num_output_channels=model_init.shape[1], \
            num_channels_down = [128, 128, 128, 128, 128], \
            num_channels_up =   [128, 128, 128, 128, 128],
            num_channels_skip =    [4, 4, 4, 4, 4], 
            filter_size_up = 3,filter_size_down = 3,filter_skip_size=1, \
            upsample_mode='bilinear', # downsample_mode='avg',
            need1x1_up=True,need_softmax=False, \
            need_sigmoid=False, need_bias=True, pad=pad, \
            act_fun='LeakyReLU')

netG = netG.to(device)

"""Compute number of parameters"""
s  = sum(np.prod(list(p.size())) for p in netG.parameters())
print ('Number of netG params: %d' % s)

### get input of network
net_input = get_noise(input_num=1,input_depth=k_channels,
                      method=INPUT,spatial_size=model_init.shape[2:]).to(device)

### define the loss
if dip_pretrain_loss == 'L1':
    criterion = torch.nn.L1Loss()
elif dip_pretrain_loss == 'L2':
    criterion = torch.nn.MSELoss()
else:
    raise NotImplementedError


### define the result path
dip_pretrain_result = ResultPath+'DIP_pretrain'+'_inchannel'+str(k_channels)+'_upsample'+str(upsample_times)+'_loss'+str(dip_pretrain_loss)+'_lr'+str(dip_pretrain_lr)+ '_ite'+str(pretrain_num_iter) 

if reg_noise_std >0:
    dip_pretrain_result = dip_pretrain_result+'_reg_std'+str(reg_noise_std)
    
if LR_decrease:
    dip_pretrain_result = dip_pretrain_result+ \
    '_step'+str(OptiScStepSize)+'_gamma'+str(OptiScGamma)
    
dip_pretrain_result = dip_pretrain_result+'/'

if not os.path.exists(dip_pretrain_result):    
    os.makedirs(dip_pretrain_result) 
    
######  Training the Network ######
print() 
print('*******************************************') 
print('*******************************************') 
print('             START TRAINING                     ') 
print('*******************************************') 
print('*******************************************') 

start_time = time.time()
i = 0
Loss = 0.0

def closure():
    global i,net_input_saved,noise,Loss
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    else:
        net_input = net_input_saved
        
    out = netG(net_input)
   
    loss = criterion(out,model_init)
    Loss = np.append(Loss,loss.item())
    loss.backward()
           
    i += 1
    
    if i % 300 ==0:
        # plot Loss               
        PlotDIPLoss(loss=Loss, SaveFigPath=dip_pretrain_result)
        
    return Loss,net_input

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, netG, net_input)
optimizer,Loss,net_input=optimize(OPTIMIZER, p, closure, \
                   dip_pretrain_lr,pretrain_num_iter, \
                   LR_decrease,OptiScStepSize,OptiScGamma)

out = netG(net_input)
out, model_init = out.reshape(nz,ny), model_init.reshape(nz,ny)


snr = ComputeSNR(out.detach().cpu().numpy(),model_init.detach().cpu().numpy())
ssim = ComputeSSIM(out.detach().cpu().numpy(),model_init.detach().cpu().numpy())
rerror = ComputeRE(out.detach().cpu().numpy(),model_init.detach().cpu().numpy())

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m  {:.0f}s' .format(time_elapsed //60 , time_elapsed % 60))
# record the consuming time
np.savetxt(dip_pretrain_result+'dip_result.txt', np.hstack((pretrain_num_iter,time_elapsed//60,time_elapsed % 60,snr,ssim,rerror)), fmt='%3.4f')
 
### save the output       
spio.savemat(dip_pretrain_result+str(dataname)+'_dip_pretrain_rec.mat', \
                             {'rec':out.cpu().data.numpy(),
                              'net_input':net_input.cpu().data.numpy(),
                             'Loss':Loss})

### save the pretrained network
state = {'ite': pretrain_num_iter,
             'state_dict': netG.state_dict(),
             'optim': optimizer,
            }                        
torch.save(state, dip_pretrain_result +'net_ite'+str(pretrain_num_iter) + ".pth.tar") 
            