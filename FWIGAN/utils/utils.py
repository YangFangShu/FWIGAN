"""
Common function
 

@author: fangshuyang (yangfs@hit.edu.cn)

"""

import torch
import deepwave
import numpy as np
import scipy
import scipy.io as spio
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import math
from math import exp
from IPython.core.debugger import set_trace
import scipy.stats
import warnings
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)

from scipy.signal import sosfilt
from scipy.signal import zpk2sos
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter,hanning
from ParamConfig import *


def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency.
    
    Args:
        freq: A float specifying the central frequency of the wavelet
        length: An integer specifying the number of time steps to use
        dt: A float specifying the time interval between time steps
        peak_time: A float specifying the time (in time units) at which the
                   peak amplitude of the wavelet occurs

    Returns:
        A 1D Numpy array of length 'length' containing a Ricker wavelet
    """
    t = (np.arange(length) * dt - peak_time).astype(np.float32)
    y = (1 - 2 * np.pi**2 * freq**2 * t**2) \
            * np.exp(-np.pi**2 * freq**2 * t**2)
    return y


def createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, num_dims,source_spacing, receiver_spacing,source_depth,receiver_depth):
    """
        Create arrays containing the source and receiver locations
        Args:
            num_shots: nunmber of shots
            num_sources_per_shot: number of sources per shot
            num_receivers_per_shot： number of receivers per shot
            num_dims: dimension of velocity model
        return:
            x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
            x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions] 
    """    
    x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    if source_depth != 0:
        x_s[:, 0, 0] = source_depth        
    x_s[:, 0, 1] = torch.arange(1,num_shots+1).float() * source_spacing
    x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    if receiver_depth != 0:
        x_r[:, :, 0] = receiver_depth
    x_r[0, :, 1] = torch.arange(1,num_receivers_per_shot+1).float() * receiver_spacing
    x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

    return x_s, x_r


def createSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots, num_sources_per_shot):
    """
        Create true source amplitudes [nt, num_shots, num_sources_per_shot]
        This is implemented by numpy
        Args:
            peak_freq : frequency for source
            peak_source_time: delay

        return:
            source_amplitudes

    """

    source_amplitudes_true = np.tile(ricker(peak_freq, nt, dt, peak_source_time).reshape(-1, 1, 1),[1,num_shots, num_sources_per_shot])
    
    
    return source_amplitudes_true



def createInitialModel(model_true, gfsigma, lipar, fix_value_depth, device):
    """
        Create 2D initial guess model for inversion ('line','lineminmax','const','GS')
    """
    assert gfsigma in ['line','lineminmax','constant','GS']
    model_true = model_true.cpu().detach().numpy()   
    shape = model_true.shape
    if fix_value_depth > 0:
        const_value = model_true[:fix_value_depth,:]
   
    if gfsigma == 'line':
    # generate the line increased initial model
        value = np.linspace(model_true[fix_value_depth,np.int(shape[1]/2)], \
                            model_true[-1,np.int(shape[1]/2)]*lipar,num=shape[0]-fix_value_depth, \
                            endpoint=True,dtype=float).reshape(-1,1)
        value = value.repeat(shape[1],axis=1)        
    elif gfsigma == 'lineminmax':
    # generate the line increased initial model (different min/max value)
        value = np.linspace(model_true.min()*lipar, \
                            model_true.max(),num=shape[0]-fix_value_depth,
                            endpoint=True,dtype=float).reshape(-1,1)
        
        value = value.repeat(shape[1],axis=1)      
    elif gfsigma == 'const':
    # generate the constant initial model
        value = model_true[fix_value_depth, int(np.floor(shape[1] / 2))] * np.ones(shape[0]-fix_value_depth,shape[1])
    # generate the initial model by using Gaussian smoothed function
    else:
        value = scipy.ndimage.gaussian_filter(model_true[fix_value_depth:,:], sigma=gfsigma)
        
    if fix_value_depth > 0:
        model_init = np.concatenate([const_value,value],axis=0)
    else:
        model_init = value
        
    model_init = torch.tensor(model_init)
    # Make a copy so at the end we can see how far we came from the initial model
    model = model_init.clone()
   
    model = model.to(device)
    # set the requires_grad to True to update the model
    model.requires_grad = True
    print('model size:',model.size())
    
    return model, model_init


def createdata(model,dx,source_amplitudes,x_s,x_r,dt, \
               pml_width,order,survey_pad,device):
    """
        Create data depends on the velocity model 
    """
    prop = deepwave.scalar.Propagator({'vp': model.to(device)},dx,pml_width, \
                                      order,survey_pad)
    # the shape of receiver_amplitudes is [nt, num_shots, num_receivers_per_shot]
    receiver_amplitudes = prop(source_amplitudes.to(device), \
                               x_s.to(device), \
                               x_r.to(device),dt).cpu()

    return receiver_amplitudes
   

def createFilterSourceAmp(peak_freq,nt,dt,peak_source_time,num_shots, \
                          num_sources_per_shot,use_filter,filter_type, \
                          freqmin,freqmax,corners,df):
    """
        Create source amplitudes with filter function
        Args:
            peak_freq : frequency for source
            peak_source_time: delay
            filt_type: type of filter ('highpass','lowpass','bandpass')

        return:
            source_amplitudes_filt

    """

    source_amplitudes = ricker(peak_freq, nt, dt, peak_source_time)
    if use_filter:
        filt_data = seismic_filter(data=source_amplitudes, \
                           filter_type=filter_type,freqmin=freqmin, \
                           freqmax=freqmax,df=df,corners=corners)
        filt_data = filt_data
    else:
        filt_data = source_amplitudes
        
    source_amplitudes_filt = np.tile(filt_data.reshape(-1,1,1),[1, num_shots, num_sources_per_shot])
    return source_amplitudes_filt


def seismic_filter(data,filter_type,freqmin,freqmax,df,corners,zerophase=False,axis=-1):
    """
    create the fileter for removing the frequency component of seismic data 
    """
    assert filter_type.lower() in ['bandpass', 'lowpass', 'highpass']

    if filter_type == 'bandpass':
        if freqmin and freqmax and df:
            filt_data = bandpass(data, freqmin, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'lowpass':
        if freqmax and df:
            filt_data = lowpass(data, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'highpass':
        if freqmin and df:
            filt_data = highpass(data, freqmin, df, corners, zerophase, axis)
        else:
            raise ValueError
    return filt_data



    
def bandpass(data, freqmin, freqmax, df, corners, zerophase, axis):
    """
    Butterworth-Bandpass Filter.
    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)

    
def lowpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Lowpass Filter.
    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)


def highpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Highpass Filter.
    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).
    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)



def loadtruemodel(data_dir, num_dims, vmodel_dim):
    """
        Load the true model
    """
    
    if num_dims != len(vmodel_dim.reshape(-1)):
        raise Exception('Please check the size of model_true!!')
    # prefer the depth direction first, that is the shape is `[nz, (ny, (nx))]`
    if num_dims == 2:       
        model_true = (np.fromfile(data_dir, np.float32).reshape(vmodel_dim[1],vmodel_dim[0]))
        model_true = np.transpose(model_true,(1,0)) # I prefer having depth direction first
    else:
        raise Exception('Please check the size of model_true!!')
   
    model_true = torch.Tensor(model_true) # Convert to a PyTorch Tensor
    
    return model_true

def loadrcv(rcvfile,device):
    """
        Load the receiver amplitude
    """
    data_mat     = spio.loadmat(rcvfile)    
   
    receiver_amplitudes_true  = torch.from_numpy(np.float32(data_mat[str('true')]))
    
    receiver_amplitudes_true = receiver_amplitudes_true.to(device)
    return receiver_amplitudes_true 




def loadinitmodel(initfile,device):
    """
        Load initial model guess
    """
    model_mat = spio.loadmat(initfile)
    model_init = torch.from_numpy(np.float32(model_mat[str('initmodel')]))
    model =  model_init.clone().to(device)
    model.requires_grad = True 
    
    return model, model_init

def fix_model_grad(fix_value_depth,model):
    assert fix_value_depth>0
    device = model.device
    # Create Gradient mask
    gradient_mask = torch.zeros(model.shape).to(device)
    # set the [receiver_depth:,:] = 1; [:receiver_depth,:] = 0;
    gradient_mask[fix_value_depth:,:] = 1.0
    # only update the [receiver_depth:,:]
    model.register_hook(lambda grad: grad.mul_(gradient_mask))

def loadinitsource(initsafile,device):
    """
        Load initial source amplitude guess
    """
    source_mat = spio.loadmat(initsafile)
    source_init = torch.from_numpy(np.float32(source_mat[str('initsource')])).to(device)
    source_true =  torch.from_numpy(np.float32(source_mat[str('truesource')])).to(device)
       
    return source_init, source_true


    
def createlearnSNR(init_snr_guess,device):
    """
        create learned snr when amplitude is noisy and try to learn the noise
    """
    learn_snr_init = torch.tensor(init_snr_guess)
    learn_snr = learn_snr_init.clone()
    learn_snr = learn_snr.to(device)
    #set_trace()
    learn_snr.requires_grad = True
    
    return learn_snr, learn_snr_init
      

    
    
def gaussian(window_size, sigma):
    """
    gaussian filter
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    create the window for computing the SSIM
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def ComputeSSIM(img1, img2, window_size=11, size_average=True):
    """
    compute the SSIM between img1 and img2
    """
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    
    if len(img1.size()) == 2:
        d = img1.size()
        img1 = img1.view(1,1,d[0],d[1])
        img2 = img2.view(1,1,d[0],d[1])
    elif len(img1.size()) == 3:
        d = img1.size()
        img1 = img1.view(d[2],1,d[0],d[1])
        img2 = img2.view(d[2],1,d[0],d[1]) 
    else:
        raise Exception('The shape of image is wrong!!!')
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ComputeSNR(rec,target):
    """
       Calculate the SNR between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    snr = 0.0
    if len(rec.shape) == 3:
        for i in range(rec.shape[0]):
            rec_ind     = rec[i,:,:].reshape(np.size(rec[i,:,:]))
            target_ind  = target[i,:,:].reshape(np.size(rec_ind))
            s      = 10*np.log10(sum(target_ind**2)/sum((rec_ind-target_ind)**2))
            snr    = snr + s
        snr = snr/rec.shape[0]
    elif len(rec.shape) == 2:
        rec       = rec.reshape(np.size(rec))
        target    = target.reshape(np.size(rec))
        snr       = 10*np.log10(sum(target**2)/sum((rec-target)**2))
    else:
        raise Exception('Please reshape the Rec to correct Dimension!!')
    return snr

def ComputeRSNR(rec,target):
    """
       Calculate the regressed-SNR(RSNR) between reconstructed image and true  image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    rec_ind     = rec.reshape(np.size(rec))
    target_ind  = target.reshape(np.size(rec))
    slope,intercept, _, _, _ = scipy.stats.linregress(rec_ind,target_ind)
    r           = slope*rec_ind + intercept
    rsnr        = 10*np.log10(sum(target_ind**2)/sum((r-target_ind)**2))
    
    if len(rec.shape) == 2:
        rec  = r.reshape(rec.shape[0],rec.shape[1])
    elif len(rec.shape) == 3:
        rec  = r.reshape(rec.shape[0],rec.shape[1],rec.shape[2])
    else:
        raise Exception('Wrong shape of reconstruction!!!')
    return rsnr, rec

def ComputeRE(rec,target):
    """
    Compute relative error between the rec and target
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
       
    rec    = rec.reshape(np.size(rec))
    target = target.reshape(np.size(rec))
    rerror = np.sqrt(sum((target-rec)**2)) / np.sqrt(sum(target**2))
    
    return rerror



def AddAWGN(data, snr):
    """
       Add additive white Gaussian noise to data such that the SNR is snr
    """
    if len(data.size()) !=3:
        assert False, 'Please check the data shape!!!'
    
    # change the shape to [num_shots,nt,num_receiver]
    data1 = data.permute(1,0,2)
    dim = data1.size() 
    device = data1.device
    SNR = snr
    y_noisy = data1 + torch.randn(dim).to(device)*(torch.sqrt(torch.mean((data1.detach()**2).reshape(dim[0],-1),dim=1)/(10**(SNR/10)))).reshape(dim[0],1,1).repeat(1,dim[1],dim[2])
    
    # change the shape to [nt,num_shots,num_receiver]
    y_noisy = y_noisy.permute(1,0,2)
                       
    # check the shape of y_noisy is equal to data or not
    if y_noisy.size() != data.size():
        assert False, 'Wrong shape of noisy data!!!'                 
  
    return y_noisy


def TVLoss(x):
    """Compute TV loss for an image x
        Args:
            x: image, torch.Variable of torch.Tensor
        Returns:
            tv loss
     """
    x      = x.float()
    dh     = torch.pow(x[:,1:] - x[:,:-1], 2)
    dw     = torch.pow(x[1:,:] - x[:-1,:], 2)
    tvloss = torch.sum(torch.pow(dh[:-1,:] + dw[:, :-1], 0.5)).float()

    return tvloss



def ATVLoss(x):    
    """Compute L1-based anisotropic TV loss for x

    Args:
        x: image, torch.Variable of torch.Tensor
    Returns:
           ATV loss
    """
    x        = x.float()
    dh       = x[:,1:] - x[:,:-1]
    dw       = x[1:,:] - x[:-1,:]
    atvloss  = torch.sum(torch.abs(dh[:-1,:]) + torch.abs(dw[:, :-1])).float()

    return atvloss

def updateinput(net_input_saved,noise,i,reg_noise_std,reg_noise_decayevery):
    """
    update the input of decoder
    """
    
    if reg_noise_decayevery !=0 and i % reg_noise_decayevery == 0:
        reg_noise_std *= 0.7
    net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        
    return net_input

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_num,input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [input_num, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params