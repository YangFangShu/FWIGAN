#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fangshuyang (yangfs@hit.edu.cn)

#############  FWI via DIP （FWIDIP) ##################

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
    
### Load true velocity model and create constant model ###
model_true = loadtruemodel(data_path, num_dims, vmodel_dim).to(device)

### Create arrays containing the source and receiver locations ###
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims, source_spacing, receiver_spacing,source_depth,receiver_depth)
x_s, x_r = x_s.to(device), x_r.to(device)


### Load initial source amplitude for inversion ###
initsafile = ResultPath+str(dataname)+'_initsource.mat'
print('Loading initial source amplitude...')
source_amplitudes_init, source_amplitudes_true = loadinitsource(initsafile,device) 


### Load  initial model guess ###
initfile = ResultPath+str(dataname)+'_initmodel.mat'
print('Loading initial model...')
model, model_init = loadinitmodel(initfile,device) 
# set model as parameter to be updated
model = torch.nn.Parameter(model)

### Load the pretrained generative network  #############
if use_dip:
    netG = skip(k_channels, output_channels, 
               num_channels_down = [128, 128, 128, 128, 128],
               num_channels_up =   [128, 128, 128, 128, 128],
               num_channels_skip =    [4, 4, 4, 4, 4],  
               filter_size_up = 3,filter_size_down = 3,filter_skip_size=1,
               upsample_mode='bilinear', # downsample_mode='avg',
               need1x1_up=True,need_softmax=False,
               need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')
else:
    assert False,"Please choose the correct generator!!!"
    
    
dip_pretrain_result = ResultPath+'DIP_pretrain'+'_inchannel'+str(k_channels)+'_upsample'+str(upsample_times)+'_loss'+str(dip_pretrain_loss)+'_lr'+str(dip_pretrain_lr)+ '_ite'+str(pretrain_num_iter) 

if reg_noise_std >0:
    dip_pretrain_result = dip_pretrain_result+'_reg_std'+str(reg_noise_std)
    
if LR_decrease:
    dip_pretrain_result = dip_pretrain_result+ \
    '_step'+str(OptiScStepSize)+'_gamma'+str(OptiScGamma)
    
dip_pretrain_result = dip_pretrain_result+'/'

model_file  = dip_pretrain_result+'net_ite'+str(pretrain_num_iter)+'.pth.tar'
state = torch.load(model_file)
netG.load_state_dict(state['state_dict'])
netG = netG.to(device)

### load the saved  receiver_amplitudes_true ###
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':   
        ResultPath = ResultPath+'AWGN_var'+str(noise_var)+'/'
    
    noisercvfile = ResultPath+str(dataname)+'_noisercv_amps.mat'
    print('Loading noisy receiver amplitudes...')
    rcv_amps_true = loadrcv(noisercvfile,device)   
else:   
    rcvfile = ResultPath+str(dataname)+'_rcv_amps.mat'
    print('Loading clean receiver amplitudes...')
    rcv_amps_true = loadrcv(rcvfile,device)
# Verify that the returned receiver amplitudes have shape

print('receiver amplitude shape',rcv_amps_true.shape)

### generate the random input  ###
net_input = get_noise(input_num=1,input_depth=k_channels,
                      method=INPUT,spatial_size=model_true.shape).to(device)

noise = net_input.clone().to(device)
net_input_saved = net_input.clone().to(device)

# optimizer and scheduler
optimizer = optim.Adam(netG.parameters(),lr=dip_v_lr,betas=(0.5,0.9), eps=1e-8,weight_decay=0)

if dip_weight_decay>0:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                step_size=dip_stepsize, \
                                                gamma=dip_weight_decay)   
      
## loss option: L1, L2,  1-D W1
if fwi_loss_type == 'L1':
    criterion = torch.nn.L1Loss()
elif fwi_loss_type == 'L2':
    criterion = torch.nn.MSELoss() 
## transform type for 1-D W1
elif fwi_loss_type == 'W1':
    trans_type = 'linear' # linear, square, exp, softplus, abs
else:
    raise NotImplementedError
    
### define the result path
dip_result = ResultPath+'DIP'+'_loss'+str(fwi_loss_type)+'_lr'+str(dip_v_lr)+'_batch'+str(dip_num_batches)+'_norm'+str(dip_data_norm)+ '_epoch'+str(dip_num_epochs)

if dip_weight_decay>0:
    dip_result = dip_result+'_dra'+str(dip_weight_decay)+'_step'+str(dip_stepsize)

    
dip_result = dip_result+'/'
    
if not os.path.exists(dip_result):    
    os.makedirs(dip_result)    

######  Training the Network ######
print() 
print('*******************************************') 
print('               START FWIDIP                       ') 
print('*******************************************') 

############## Init  ################# 
SNR = 0.0
RSNR = 0.0
SSIM = 0.0
Loss = 0.0
ERROR = 0.0
##################### Mian loop ####################
def fwi_dip_main():
    
    global model_true, SNR, RSNR, SSIM, Loss, ERROR,net_input_saved,noise 
        
    t_start = time.time()
    vmin, vmax = np.percentile(model_true.detach().cpu().numpy(), [2,98]) # For plotting   
    model_true = model_true.view(nz,ny)
    # number of shots per batch (the interval is num_batches)
    num_shots_per_batch = int(num_shots / dip_num_batches) 

    for epoch in range(dip_num_epochs):
        epoch_loss = 0.0
        for it in range(dip_num_batches):
            iteration = epoch*dip_num_batches+it+1
            netG.train()
            optimizer.zero_grad()
            if reg_noise_std > 0:
                net_input = updateinput(net_input_saved,noise,iteration,reg_noise_std,reg_noise_decayevery=500)
            else:
                net_input = net_input_saved
            out = netG(net_input)
            model = out.reshape(nz,ny)
            
            prop = deepwave.scalar.Propagator({'vp': model}, dx, pml_width, \
                                              order,survey_pad)
            batch_src_amps = source_amplitudes_init.repeat(1, num_shots_per_batch, 1)
            batch_rcv_amps_true = rcv_amps_true[:,it::dip_num_batches].to(device)
            
            batch_x_s = x_s[it::dip_num_batches].to(device)
            batch_x_r = x_r[it::dip_num_batches].to(device)
            batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
            
            if fwi_loss_type == 'L1' or fwi_loss_type == 'L2':             
                if dip_data_norm:
                    # normalize the amplitude of each shot in the range of [-1 1]
                    batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*fscale
                    batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*fscale
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)            
                else:
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
            elif fwi_loss_type == 'W1':
                loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true,trans_type,theta=1.1)         
            else:
                raise NotImplementedError
                
            epoch_loss += loss.item()

            loss.backward()
            
            # Clips gradient value of model, source ampliude
            #torch.nn.utils.clip_grad_value_(netG.parameters(),1e3) 
            
            optimizer.step()
            
            # clip the model value that keep the min value is larger than 0
            model.data=torch.clamp(model.data,min=1e-12)
              
            if dip_weight_decay>0:        
                scheduler.step()
            
            # compute the SNR, RSNR, SSIM and RE between GT and inverted model
            snr = ComputeSNR(model.detach().cpu().numpy(), \
                      model_true.detach().cpu().numpy())
            SNR = np.append(SNR, snr)

            ssim = ComputeSSIM(model.detach().cpu().numpy(), \
                      model_true.detach().cpu().numpy())
            SSIM = np.append(SSIM, ssim)
           
            rerror = ComputeRE(model.detach().cpu().numpy(), \
                      model_true.detach().cpu().numpy())
            ERROR = np.append(ERROR, rerror)
             
        print('Epoch:', epoch+1, 'Loss: ', epoch_loss / dip_num_batches)
        Loss = np.append(Loss, epoch_loss / dip_num_batches)
        
        
        if iteration % plot_ite== 0:
            with torch.no_grad():
                out = netG(net_input)
                model = out.reshape(nz,ny)
            plotcomparison(gt=model_true.cpu().data.numpy(), \
                        pre=model.cpu().data.numpy(), \
                        ite=iteration,SaveFigPath=dip_result) 
            # plot Loss               
            PlotFWILoss(loss=Loss, SaveFigPath=dip_result)

            # plot SNR, RSNR, and SSIM
            PlotSNR(SNR=SNR, SaveFigPath=dip_result)
            #PlotRSNR(RSNR=RSNR, SaveFigPath=dip_result)
            PlotSSIM(SSIM=SSIM, SaveFigPath=dip_result)
            PlotERROR(ERROR=ERROR,SaveFigPath=dip_result)
        
        
        if (epoch+1) % savepoch ==0 or (epoch+1) == dip_num_epochs:
        #####------------------Save Rec and Loss-----------------#####
            with torch.no_grad():
                out = netG(net_input)
                model = out.reshape(nz,ny)
            spio.savemat(dip_result+'DIPRec'+'.mat', \
                             {'rec':model.cpu().data.numpy(),
                             'net_input':net_input.cpu().data.numpy()}) 
            spio.savemat(dip_result+'DIPMetric'+'.mat', \
                             {'SNR':SNR,'SSIM':SSIM,'Loss':Loss,'ERROR':ERROR})
    
    
    t_end = time.time()
    elapsed_time = t_end - t_start
    print('Running complete in {:.0f}m  {:.0f}s' .format(elapsed_time //60 , elapsed_time % 60))
    np.savetxt(dip_result+'run_result.txt', np.hstack((epoch+1,elapsed_time//60,elapsed_time % 60,snr,ssim,rerror)), fmt='%5.4f') #ssim,
    
#####################  Run Code ####################    
if __name__ == "__main__":
    fwi_dip_main()
    exit(0)