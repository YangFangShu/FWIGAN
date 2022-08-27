#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fangshuyang (yangfs@hit.edu.cn)

#########       FWI via WGAN-GP       ############

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


### Create arrays containing the source and receiver locations ###
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims, source_spacing, receiver_spacing,source_depth,receiver_depth)
x_s, x_r = x_s.to(device), x_r.to(device)


### Load initial guess source amplitude for inversion ###
initsafile = ResultPath+str(dataname)+'_initsource.mat'
print('Loading initial source amplitude...')
source_amplitudes_init, source_amplitudes_true = loadinitsource(initsafile,device) 


### Load  initial model guess ###
initfile = ResultPath+str(dataname)+'_initmodel.mat'
print('Loading initial model...')
model, model_init = loadinitmodel(initfile,device) 
# set model as parameter to be updated
model = torch.nn.Parameter(model)


### load the saved  receiver_amplitudes_true ###
if AddNoise == True and noise_var != None:
    ### define the path and file name of noisy rcv ###
    if noise_type == 'Gaussian':   
        ResultPath = ResultPath+'AWGN_var'+str(noise_var)+'/'
    
    noisercvfile = ResultPath+str(dataname)+'_noisercv_amps.mat'
    print('Loading noisy receiver amplitudes...')
    rcv_amps_true = loadrcv(noisercvfile,device)   
else:   
    rcvfile = ResultPath+str(dataname)+'_rcv_amps.mat'
    print('Loading clean receiver amplitudes...')
    #rcv_amps_true_norm = loadFWIrcv(recfile,device)
    rcv_amps_true = loadrcv(rcvfile,device)
# Verify that the returned receiver amplitudes have shape [nt, num_shots, num_receivers_per_shot]
print('receiver amplitude shape',rcv_amps_true.shape)


###  Define the PhysicsGenerator and Discriminator  ###
PhySimulator = PhySimulator(dx,num_shots,gan_num_batches, \
                            x_s,x_r,dt,pml_width, \
                            order,survey_pad).to(device)

### Discriminator is 6 Conv + 2 fully-connected layers network ###
num_shots_per_batch = int(num_shots / gan_num_batches) # number of shots per batch (the interval is num_batches)

Filters = np.array([DFilter,2*DFilter,4*DFilter,8*DFilter,16*DFilter,32*DFilter],dtype=int)

"""
note that you need to change the dimension in netD when the size of amplitude changes
"""
netD = Discriminator(batch_size=num_shots_per_batch,ImagDim=[nt,num_receivers_per_shot],
                     LReLuRatio=0.1,filters=Filters,
                     leak_value=leak_value)  

### init the hyper-parameters of netD ###
netD.apply(lambda m: weights_init(m, leak_value))
netD = netD.to(device)

### Optimizer setting for discriminator ###
optim_d = optim.Adam(netD.parameters(),lr=gan_d_lr,betas=(0.5, 0.9), \
                    eps=1e-8, weight_decay=0)

"""Compute number of parameters"""
s  = sum(np.prod(list(p.size())) for p in netD.parameters())
print ('Number of netD params: %d' % s)


### Optimizer setting for model  ###
optim_g = optim.Adam([{'params' : model, 'lr':gan_v_lr, 'betas':(0.5, 0.9), 'eps':1e-8, 'weight_decay':0}])

### Scheduler for optimizer ###
if gan_weight_decay>0: 
    scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, \
                                                step_size=gan_stepsize, \
                                                gamma=gan_weight_decay)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, \
                                                step_size=gan_stepsize, \
                                                gamma=gan_weight_decay)
    
    
# try to learn the Gaussian noise when the amplitude is noisy    
if AddNoise == True and noise_var != None and learnAWGN == True:
    init_snr_guess = np.array(20,dtype="float32")
    gan_noi_lr = 1e-2
    learn_snr, learn_snr_init = createlearnSNR(init_snr_guess,device)
    learn_snr = torch.nn.Parameter(learn_snr)
    optim_noi = optim.Adam([{'params':learn_snr,'lr':gan_noi_lr,'betas':(0.5, 0.9),'eps':1e-8,'weight_decay':0}])
    if gan_weight_decay>0:    
        scheduler_noi =torch.optim.lr_scheduler.StepLR(optim_noi, \
                                                step_size=gan_stepsize, \
                                                gamma=gan_weight_decay)        
else:
    learn_snr = None 
 
### define the result path   
gan_result = ResultPath+'GAN'+'_vlr'+str(gan_v_lr)+'_dlr'+str(gan_d_lr)+'_batch'+str(gan_num_batches)+'_epoch'+str(gan_num_epochs)+'_criIte'+str(criticIter)

if gan_weight_decay>0:
    gan_result = gan_result+'_dra'+str(gan_weight_decay)+'_step'+str(gan_stepsize)

if lamb != 10:
    gan_result = gan_result+'_lamb'+str(lamb)

if AddNoise == True and noise_var != None and learnAWGN == True:
    gan_result = gan_result+'_noilr'+str(gan_noi_lr)+'_initsnr'+str(init_snr_guess)
    
gan_result = gan_result+'/'

if not os.path.exists(gan_result):
    os.makedirs(gan_result)


######  Training the Network ######
print() 
print('*******************************************') 
print('*******************************************') 
print('             START FWIGAN                       ') 
print('*******************************************') 
print('*******************************************') 

############## Initialization  ################# 
one = torch.tensor(1, dtype=torch.float) 
mone = one * -1
one = one.to(device)
mone = mone.to(device)
DLoss  = 0.0
WDist  = 0.0
GLoss  = 0.0
DiscReal = 0.0
DiscFake = 0.0
SNR = 0.0
SSIM = 0.0
ERROR = 0.0

##################### Mian loop ####################
def fwi_gan_main():
    
    global DLoss, WDist,GLoss, one, mone, DiscReal, DiscFake, model_true, SNR, SSIM, ERROR 
        
    start_time = time.time()
    vmin, vmax = np.percentile(model_true.detach().cpu().numpy(), [2,98]) 
    model_true = model_true.view(nz,ny)
        
    for epoch in range(start_epoch, gan_num_epochs):
        print("Epoch: " + str(epoch+1))
        
        for it in range(gan_num_batches):
            iteration = epoch*gan_num_batches+it+1
            #####-------------------(1)--TRAIN D------------------------#####
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad_(True)    # they are set to False below in training G
            for j in range(criticIter):
                # set netD in training stage
                netD.train()                
                netD.zero_grad()                
               
                # for inner loop training of Discirminator 
                if it*criticIter+j < gan_num_batches:
                    # the true receiver amplitude(e.g. shot num: 0,10,20,30,40,50)
                    batch_rcv_amps_true = rcv_amps_true[:,it*criticIter+j::gan_num_batches]
                else:
                    # take the shots per batch from starting
                    batch_rcv_amps_true = rcv_amps_true[:,((it*criticIter+j) % gan_num_batches)::gan_num_batches]
               
                d_real = batch_rcv_amps_true.detach()
                with torch.no_grad():                    
                    model_fake = model  # totally freeze G, training D 
                    if AddNoise == True and noise_var != None and learnAWGN == True:
                        learn_snr_fake = learn_snr.detach()
                    else:
                        learn_snr_fake = None
                ### obtain the simulated data from current model
      
                d_fake = PhySimulator(model_fake, \
                                      source_amplitudes_init,it,criticIter,j, \
                                      'TD',AddNoise,learn_snr_fake,learnAWGN)
                
                ### save the first d_fake and d_real sample
                if j == criticIter-1 and iteration % plot_ite ==0:
                     plotfakereal(fakeamp=d_fake[:,0].cpu().data.numpy(), \
                          realamp=d_real[:,0].cpu().data.numpy(), \
                              ite=iteration,cite=j+1,SaveFigPath=gan_result)
                
                # train with real data
                # change the dim from [nt,num_shots,nnum_receiver] to [num_shots,nt,num_receiver]
                d_real = d_real.permute(1,0,2)
                # insert one dim at the second place, so that the dim is [num_shots,1,nt,num_receiver]
                d_real = d_real.unsqueeze(1)
                disc_real = netD(d_real)
                disc_real = disc_real.mean()
    
                # train with fake data
                d_fake = d_fake.permute(1,0,2)
                d_fake = d_fake.unsqueeze(1)
                disc_fake = netD(d_fake)
                disc_fake = disc_fake.mean()
    
                DiscReal = np.append(DiscReal, disc_real.item())
                DiscFake = np.append(DiscFake, disc_fake.item())
            
                # train with interpolates data                
                gradient_penalty = calc_gradient_penalty(netD,d_real,d_fake, \
                                                     batch_size=num_shots_per_batch, \
                                                     channel=1,lamb=lamb, \
                                                     device=device)
                             
                disc_cost = disc_fake - disc_real + gradient_penalty
                print ('Epoch: %03d  Ite: %05d  DLoss: %f' % (epoch+1,iteration,disc_cost.item()))
                
                w_dist = -(disc_fake  - disc_real)
                
                DLoss  = np.append(DLoss, disc_cost.item())
                WDist  = np.append(WDist, w_dist.item())
                
                disc_cost.backward()
                
                # Clips gradient norm of netD parameters
                torch.nn.utils.clip_grad_norm_(netD.parameters(),1e3) #1e3 for smoothed initial model; 1e6 for linear model
                
                # optimize
                optim_d.step()
                
                # scheduler for discriminator      
                if gan_weight_decay>0:        
                    scheduler_d.step()
                    
                #####------------------VISUALIZATION---------------######
                if j ==criticIter-1 and iteration % plot_ite ==0:
                    PlotDLoss(dloss=DLoss,wdist=WDist, \
                              SaveFigPath=gan_result)                       
        
            
            #####------------(2)--Update model--------------#####
            for p in netD.parameters():
                p.requires_grad_(False)  # freeze D,to avoid computation
    
            gen_cost = None
            for k in range(1):                
                
                optim_g.zero_grad()
                if AddNoise == True and noise_var != None and learnAWGN == True:
                    optim_noi.zero_grad() 
                   
                ### generate the g_fake
                g_fake = PhySimulator(model, \
                                      source_amplitudes_init,it,criticIter,j, \
                                      'TG',AddNoise,learn_snr,learnAWGN)
                
                g_fake = g_fake.permute(1,0,2)
                g_fake = g_fake.unsqueeze(1)
                
                if fix_value_depth > 0:
                    fix_model_grad(fix_value_depth,model)
                
                ### compute loss
                gen_cost = netD(g_fake)                
                gen_cost = gen_cost.mean()                
                gen_cost.backward(mone)
                gen_cost = - gen_cost
                print ('Epoch: %03d  Ite: %05d  GLoss: %f' % (epoch+1,iteration,gen_cost.item()))
                GLoss = np.append(GLoss, gen_cost.item())
               
                
                # Clips gradient value of model and source_amplitudes
                torch.nn.utils.clip_grad_value_(model,1e3) #mar_smal: 1e1 smoothed initial model; 1e3 for linear model
               
                # optimize
                optim_g.step()
                
                # scheduler for generator
                if gan_weight_decay>0:        
                    scheduler_g.step()
                    
                if AddNoise == True and noise_var != None and learnAWGN == True:
                    # Clips gradient value of learn_snr if needed
                    torch.nn.utils.clip_grad_value_(learn_snr,1e1)
                    # optimize
                    optim_noi.step()
                    if gan_weight_decay>0: 
                        scheduler_noi.step()
            
                # clip the model value that keep the minimum value is larger than 0
                model.data=torch.clamp(model.data,min=1e-12)
                
                # compute the SNR, SSIM and realtive error between GT and inverted model
                snr = ComputeSNR(model.detach().cpu().numpy(), \
                          model_true.detach().cpu().numpy())
                SNR = np.append(SNR, snr)
               
                ssim = ComputeSSIM(model.detach().cpu().numpy(), \
                          model_true.detach().cpu().numpy())
                SSIM = np.append(SSIM, ssim)
                
                rerror = ComputeRE(model.detach().cpu().numpy(), \
                      model_true.detach().cpu().numpy())
                ERROR = np.append(ERROR, rerror)
           
                       
            #####---------------VISUALIZATION---------------------#####
            if iteration % plot_ite == 0:
                # plot GT and current inverted model
                plotcomparison(gt=model_true.cpu().data.numpy(), \
                            pre=model.cpu().data.numpy(), \
                            ite=iteration,SaveFigPath=gan_result) 
                # plot GLoss               
                PlotGLoss(gloss=GLoss, SaveFigPath=gan_result)
                
                # plot SNR, RSNR, SSIM and ERROR
                PlotSNR(SNR=SNR, SaveFigPath=gan_result)
                PlotSSIM(SSIM=SSIM, SaveFigPath=gan_result)
                PlotERROR(ERROR=ERROR,SaveFigPath=gan_result)
                if AddNoise == True and noise_var != None and learnAWGN == True:
                    print('learned snr:',learn_snr)
        
            
        if (epoch+1) % savepoch ==0 or (epoch+1) == gan_num_epochs:
        #####------------------Save Rec and Metric-----------------#####
            spio.savemat(gan_result+'GANRec'+'.mat', \
                             {'rec':model.cpu().data.numpy()}) 
            spio.savemat(gan_result+'GANMetric'+'.mat', \
                             {'SNR':SNR,'SSIM':SSIM,'ERROR':ERROR, \
                             'DLoss':DLoss,'WDist':WDist,'GLoss':GLoss}) 
            
       #####----------------------Save model----------------------#####
        if (epoch+1) % savepoch == 0 or (epoch+1) == gan_num_epochs:
            if gan_weight_decay>0:          
                dis_state = {
                            'epoch': epoch+1,
                            'state_dict': netD,
                            'optim_d': optim_d,
                            'scheduler_d': scheduler_d,
                            }                        
                torch.save(dis_state, gan_result + "netD.pth.tar") 
           
           
            elif gan_weight_decay == 0:
                dis_state = {
                            'epoch': epoch+1,
                            'state_dict': netD,
                            'optim_d': optim_d,
                            }                        
                torch.save(dis_state, gan_result + "netD.pth.tar") 
           
            else:
                raise NotImplementedError
                
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m  {:.0f}s' .format(time_elapsed //60 , time_elapsed % 60))
    # record the consuming time
    np.savetxt(gan_result+'run_result.txt', np.hstack((epoch+1,time_elapsed//60,time_elapsed % 60,snr,ssim,rerror)), fmt='%3.4f')
    if AddNoise == True and noise_var != None and learnAWGN == True:
        np.savetxt(gan_result+'run_result.txt', np.hstack((epoch+1,time_elapsed//60,time_elapsed % 60,snr,ssim,rerror,learn_snr.cpu().data.numpy())), fmt='%3.4f')
        

    
#####################  Run Code ####################    
if __name__ == "__main__":
    fwi_gan_main()
    exit(0)