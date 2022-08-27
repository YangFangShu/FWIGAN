#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fangshuyang (yangfs@hit.edu.cn)

#############  FWI via Deepwave (by Alan Richardson)  ##################

reference: 
   https://github.com/guoketing/deepwave-order
   https://github.com/ar4/deepwave

##########################################################################
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

### Load true velocity model  ###
model_true = loadtruemodel(data_path, num_dims, vmodel_dim).to(device)
print(' maximum value of true model: ', model_true.max())

### Load or create initial model guess ###
initfile = ResultPath+str(dataname)+'_initmodel.mat'
if os.path.exists(initfile):
    print('Initial model exists, downloading...')
    model, model_init = loadinitmodel(initfile,device)   
else:
    print('Start to generate the initial model')
    model, model_init = createInitialModel(model_true, gfsigma, lipar,fix_value_depth, device)
    vmin, vmax = np.percentile(model_true.cpu().data.numpy(), [2,98])
    plotinitmodel(model_init,model_true,vmin,vmax,ResultPath)
    spio.savemat(initfile, \
                     {'initmodel':model_init.cpu().data.numpy()}) 
# set model as parameter to be updated
model = torch.nn.Parameter(model)


### Create arrays containing the source and receiver locations ###
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims, source_spacing, receiver_spacing,source_depth,receiver_depth)
x_s, x_r = x_s.to(device), x_r.to(device)

### load or create initial source amplitude for inversion ###
initsafile = ResultPath+str(dataname)+'_initsource.mat'
if os.path.exists(initsafile):
    print('Initial source amplitude exists, downloading...')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(initsafile,device)   
else:
    ### Create true source amplitudes [nt, num_shots, num_sources_per_shot] ###
    print('Start to generate the source amplitude')
    source_amplitudes_true = createSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots, num_sources_per_shot)
    source_amplitudes_true = torch.from_numpy(source_amplitudes_true).to(device)
    if use_filter:
        source_amplitudes_filt = createFilterSourceAmp(peak_freq, nt, dt, peak_source_time, \
                                                 num_shots, num_sources_per_shot,
                                                 use_filter,filter_type,freqmin,freqmax,
                                                       corners,df)
        
        source_amplitudes_filt = torch.from_numpy(source_amplitudes_filt).to(device)
        ### get the source amplitude for simulation
        source_amplitudes_init = source_amplitudes_filt[:,0,0].reshape(-1,1,1)
    else:
        source_amplitudes_init = source_amplitudes_true[:,0,0].reshape(-1,1,1)
    
    spio.savemat(initsafile, \
                     {'initsource':source_amplitudes_init.cpu().data.numpy(),
                     'truesource':source_amplitudes_true.cpu().data.numpy()})
    
    plotinitsource(init=source_amplitudes_init.cpu().detach().numpy(), \
                   gt=source_amplitudes_true[:,0,0].cpu().detach().numpy(),SaveFigPath=ResultPath)
    
    plotsourcespectra(init_source=source_amplitudes_init.cpu().detach().numpy(),
                      true_source=source_amplitudes_true[:,0,0].cpu().detach().numpy(),
                      SaveFigPath=ResultPath)    

### define the path and file name of noise-free rcv ###
rcvfile = ResultPath+str(dataname)+'_rcv_amps.mat'
if os.path.exists(rcvfile):
    print('Clean receiver amplitudes exists, downloading......')
    receiver_amplitudes_true = loadrcv(rcvfile,device)

    ### Add additive white gaussian noise to the clean amplitude ###
    if AddNoise == True and noise_var != None:
        ### define the path and file name of noisy rcv ###
        if noise_type == 'Gaussian':   
            ResultPath = ResultPath+'AWGN_var'+str(noise_var)+'/'
        
        if not os.path.exists(ResultPath):
            os.makedirs(ResultPath) 
          
        noisercvfile = ResultPath+str(dataname)+'_noisercv_amps.mat'
        
        if os.path.exists(noisercvfile):
            print('Noisy receiver amplitudes exists, downloading......')
            # here the rcv_amps_true is contaminated by noise
            receiver_amplitudes_true = loadrcv(noisercvfile,device)
        else:
            print('Start to generate the noisy receiver amplitudes......\n',
            'Adding  %.2f  var Gaussian noise to amplitude' % (noise_var))
            receiver_amplitudes = receiver_amplitudes_true
            
            if noise_type == 'Gaussian':
                receiver_amplitudes_noise = AddAWGN(data=receiver_amplitudes.cpu().data, \
                                                    snr=noise_var).to(device)
            
            # compute the average SNR of noisy rcv
            noise_snr = ComputeSNR(receiver_amplitudes_noise.permute(1,0,2), \
                                   receiver_amplitudes.permute(1,0,2))
            print('The noise level is %.2f dB' % noise_snr)
            
            receiver_amplitudes_true = receiver_amplitudes_noise
            
            plotoneshot(receiver_amplitudes_true, ResultPath)
            # save the receiver amplitudes
            spio.savemat(noisercvfile, \
                         {'true':receiver_amplitudes_true.cpu().data.numpy(), \
                         'rcv':receiver_amplitudes.cpu().data.numpy()})          
else:
    print('Start to generate the clean receiver amplitudes......')
    
    source = source_amplitudes_init.repeat(1,num_shots, num_sources_per_shot)
        
    receiver_amplitudes = createdata(model_true,dx,source, \
                                    x_s,x_r,dt,pml_width,order, \
                                     survey_pad,device).to(device)
     
    receiver_amplitudes_true = receiver_amplitudes
    plotoneshot(receiver_amplitudes_true, ResultPath)
    # save the receiver amplitudes
    spio.savemat(rcvfile, \
                 {'true':receiver_amplitudes_true.cpu().data.numpy(), \
                 'rcv':receiver_amplitudes.cpu().data.numpy()}) 

# Verify that the returned receiver amplitudes have shape [nt, num_shots, num_receivers_per_shot]
print('receiver amplitude shape',receiver_amplitudes_true.shape)

### optimizer and scheduler
optimizer = optim.Adam([{'params': model, 'lr':fwi_lr, 'betas':(0.5, 0.9), 'eps':1e-8, 'weight_decay':0}])

if fwi_weight_decay>0: 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                      step_size=fwi_stepsize, \
                                                      gamma=fwi_weight_decay)       
### loss option: L1, L2, 1-D W1
if fwi_loss_type == 'L1':
    criterion = torch.nn.L1Loss()
elif fwi_loss_type == 'L2':
    criterion = torch.nn.MSELoss()
## transform type for 1-D W1
elif fwi_loss_type == 'W1':
    trans_type = 'linear' # linear, square, exp, softplus, abs
else:
    raise NotImplementedError


### define the  result path
fwi_result = ResultPath+'FWI'+'_loss'+str(fwi_loss_type)+'_lr'+str(fwi_lr)+ \
'_batch'+str(fwi_batch)+'_norm'+str(data_norm)+ '_epoch'+str(fwi_num_epochs)

    
if fwi_weight_decay>0:
    fwi_result = fwi_result+'_dra'+str(fwi_weight_decay)+'_step'+str(fwi_stepsize)


if AddTV:
    fwi_result = fwi_result+'_alp'+str(alpha_tv)
    
fwi_result = fwi_result+'/'
    
if not os.path.exists(fwi_result):    
    os.makedirs(fwi_result)    
        
rcv_amps_true = receiver_amplitudes_true.clone()

######  Training the Network ######
print() 
print('*******************************************') 
print('               START FWI                             ') 
print('*******************************************') 

############## Init  ################# 
SNR = 0.0
SSIM = 0.0
Loss = 0.0
ERROR = 0.0
TOL = 0.0

##################### Mian loop ####################
def fwi_main():
    
    global model_true, source_amplitudes_init, SNR, SSIM, Loss, ERROR
        
    t_start = time.time()
    vmin, vmax = np.percentile(model_true.detach().cpu().numpy(), [2,98]) # For plotting   
    model_true = model_true.view(nz,ny)
    # number of shots per batch (the interval is num_batches)
    num_shots_per_batch = int(num_shots / fwi_batch) 

    for i in range(fwi_num_epochs):
        # initialization
        epoch_loss = 0.0
        
        for it in range(fwi_batch):
            iteration = i*fwi_batch+it+1
            optimizer.zero_grad()
            prop = deepwave.scalar.Propagator({'vp': model},dx,pml_width, \
                                              order,survey_pad)
            batch_src_amps = source_amplitudes_init.repeat(1, num_shots_per_batch, 1)
            batch_rcv_amps_true = rcv_amps_true[:,it::fwi_batch].to(device)
            
            batch_x_s = x_s[it::fwi_batch].to(device)
            batch_x_r = x_r[it::fwi_batch].to(device)
            batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)
            
            if fwi_loss_type == 'L1' or fwi_loss_type == 'L2':             
                if data_norm:
                    # normalize the amplitude of each shot 
                    batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*fscale
                    batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*fscale
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)            
                else:
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
            elif fwi_loss_type == 'W1':
                loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true,trans_type,theta=1.1)            
            else:
                raise NotImplementedError
            
            if fix_value_depth > 0:
                fix_model_grad(fix_value_depth,model)
            
            epoch_loss += loss.item()
            loss.backward()
            
            # Clips gradient value of model
            torch.nn.utils.clip_grad_value_(model,1e3) 
            
            optimizer.step()
  
            # clip the model value that keep the minimum value is larger than 0
            model.data=torch.clamp(model.data,min=1e-12)
        
        if fwi_weight_decay>0:        
            scheduler.step()
            
        print('Epoch:', i+1, 'Loss: ', epoch_loss / fwi_batch)
        Loss = np.append(Loss, epoch_loss / fwi_batch)
        
        # compute the SNR,  SSIM and RE between GT and inverted model
        snr = ComputeSNR(model.detach().cpu().numpy(), \
                  model_true.detach().cpu().numpy())
        SNR = np.append(SNR, snr)
 
        ssim = ComputeSSIM(model.detach().cpu().numpy(), \
                  model_true.detach().cpu().numpy())
        SSIM = np.append(SSIM, ssim)

        rerror = ComputeRE(model.detach().cpu().numpy(), \
                  model_true.detach().cpu().numpy())
        ERROR = np.append(ERROR, rerror)
        
        if iteration % plot_ite == 0:
            plotcomparison(gt=model_true.cpu().data.numpy(), \
                        pre=model.cpu().data.numpy(), \
                        ite=iteration,SaveFigPath=fwi_result) 
            # plot Loss               
            PlotFWILoss(loss=Loss, SaveFigPath=fwi_result)

            # plot SNR, ERROR, and SSIM
            PlotSNR(SNR=SNR, SaveFigPath=fwi_result)
            PlotSSIM(SSIM=SSIM, SaveFigPath=fwi_result)
            PlotERROR(ERROR=ERROR,SaveFigPath=fwi_result)

        if (i+1) % savepoch == 0 or (i+1) == fwi_num_epochs:
        #####------------------Save Rec and Loss-----------------#####
            spio.savemat(fwi_result+'FWIRec_'+str(fwi_loss_type)+'.mat', \
                             {'rec':model.cpu().data.numpy()}) 
            spio.savemat(fwi_result+'FWIMetric_'+str(fwi_loss_type)+'.mat', \
                             {'SNR':SNR,'SSIM':SSIM, \
                              'Loss':Loss,'ERROR':ERROR}) 
                
    t_end = time.time()
    elapsed_time = t_end - t_start
    print('Running complete in {:.0f}m  {:.0f}s' .format(elapsed_time //60 , elapsed_time % 60))
    np.savetxt(fwi_result+'run_result.txt', np.hstack((fwi_num_epochs,elapsed_time//60,elapsed_time % 60,snr,ssim,rerror)), fmt='%5.4f') #ssim,
    
#####################  Run Code ####################    
if __name__ == "__main__":
    fwi_main()
    exit(0)