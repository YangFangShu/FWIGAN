"""
Common functions for plotting figures


@author: fangshuyang (yangfs@hit.edu.cn)

"""

import torch
import deepwave
import numpy as np
import scipy
import scipy.io as spio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
import math
from math import exp
from IPython.core.debugger import set_trace
import scipy.stats
import warnings  
from ParamConfig import *
from scipy.fft import fft, fftfreq 

    
    
def plotinitmodel(init,true,vmin,vmax,SaveFigPath):
    """
    plot initial velocity model
    """
    
    model_init = init
    model_true = true
    fig1, ax1 = plt.subplots(2)
    fig1.set_figheight(6)
    fig1.set_figwidth(12)
    plt.subplot(1,2,1) 
    plt.imshow(model_init.cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                   cmap='jet')
    plt.colorbar()
    plt.title('inital model')
    plt.subplot(1,2,2) 
    plt.imshow(model_true.cpu().detach().numpy(), vmin=vmin, vmax=vmax,
                   cmap='jet')
    plt.colorbar()
    plt.title('ground truth')
    plt.savefig(SaveFigPath+'init_model.png')
    plt.close()
    

def plotoneshot(receiver_amplitudes_true, SaveFigPath):
    """
        Plot one shot data
    """
    fig1, ax1 = plt.subplots()
    # note numpy.percentile(a, q, axis=None, out=None, overwrite_input=False,   interpolation='linear', keepdims=False) compute q-th percentile(s)
    # compute the 2% and 98% of summation of array 
    vmin, vmax = np.percentile(receiver_amplitudes_true[:,0].cpu().numpy(), [2,98]) 
    
    plt.imshow(receiver_amplitudes_true[:,0].cpu().numpy(), aspect='auto',
               vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar()
    plt.title('One shot data')
    
    plt.savefig(SaveFigPath+'data.png')

    plt.close()
    
def plotfakereal(fakeamp,realamp, ite, cite, SaveFigPath):
    """
        Plot one shot data of fake or real amplitude
    """
    vmin, vmax = np.percentile(realamp, [2,98]) 
    
    fig1, ax1 = plt.subplots(2)
    fig1.set_figheight(6)
    fig1.set_figwidth(12)
    
    plt.subplot(1,2,1) 
    plt.imshow(fakeamp,aspect='auto',vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar()
    plt.title('fake amplitude')
    
    plt.subplot(1,2,2) 
    plt.imshow(realamp, aspect='auto',vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar()
    plt.title('real amplitude')
    
    plt.savefig(SaveFigPath+'amp_ite{}_cite_{}.png'.format(ite,cite))

    plt.close()
    
    
def PlotFWILoss(loss, SaveFigPath):
    """
    Plot Loss of FWI
    """
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(loss[1:],color='black',lw=1,ls='-',marker='v',markersize=2,label='FWILoss') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('FWILoss')
   
    plt.savefig(str(SaveFigPath)+'FWILoss.png')
    plt.close()

def PlotDLoss(dloss,wdist,SaveFigPath):
    """
       Plot Loss for Discriminator
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(dloss[1:],color='blue',lw=1,ls='-',marker='o',markersize=2,label='DisLoss')
    line2, = ax1.plot(wdist[1:],color='green',lw=1,ls='-',marker='*',markersize=2,label='WDistance')
    
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('DLoss')
   
    plt.savefig(str(SaveFigPath)+'DLoss.png')

    plt.close()
    
    
def PlotDiscRealFake(discr,discf,SaveFigPath):
    """
       Plot DiscReal and DiscFake Loss for Discriminator
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(discr[1:],color='purple',lw=1,ls='-',marker='o',markersize=2,label='DiscReal')
    line2, = ax1.plot(discf[1:],color='green',lw=1,ls='-',marker='*',markersize=2,label='DiscFake')
    
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('Disc')
    
    plt.savefig(str(SaveFigPath)+'Disc.png')

    plt.close()
    
    
def PlotGLoss(gloss,SaveFigPath):
    """
       Plot Loss for Generator
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(gloss[1:],color='purple',lw=1,ls='-',marker='v',markersize=2,label='GenLoss') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('GLoss')
   
    plt.savefig(str(SaveFigPath)+'GLoss.png')
    plt.close()
    
def PlotDIPLoss(loss, SaveFigPath):
    """
    Plot Loss of FWIDIP
    """
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(loss[1:],color='black',lw=1,ls='-',marker='v',markersize=2,label='DIPLoss') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('DIPLoss')
   
    plt.savefig(str(SaveFigPath)+'DIPLoss.png')
    plt.close()

def plotcomparison(gt,pre,ite,SaveFigPath):
    """
    plot current inverted model and ground truth
    """
   
    dim = gt.shape
    gt = gt.reshape(dim[0],dim[1])
    pre = pre.reshape(dim[0],dim[1])
        
    vmin, vmax = np.percentile(gt, [2,98])
    fig1, ax1 = plt.subplots(2)
    fig1.set_figheight(6)
    fig1.set_figwidth(12)
    
    plt.subplot(1,2,1) 
    plt.imshow(pre,vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar()
    plt.title('inverted model')
    
    plt.subplot(1,2,2) 
    plt.imshow(gt, vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar()
    plt.title('true model')
    
    plt.savefig(SaveFigPath+'invert_ite{}.png'.format(ite))

    plt.close()
    
    
def plotinitsource(init,gt,SaveFigPath):
    """
        plot initial and true source amplitudes
    """
    t = 500
    figsize = (12, 6)
    plt.figure(figsize=figsize)
    plt.plot(init.reshape(-1).ravel()[:t], label='Initial')
    plt.plot(gt.reshape(-1).ravel()[:t], label='True')
    plt.legend()
    
    plt.title('source amplitude')
    
    plt.savefig(SaveFigPath+'source.png')

    plt.close()

    
    
def plotsourcespectra(init_source,true_source,SaveFigPath):
    """
        plot FFT frequency spectra of initial and true source amplitudes
    """
    init_source = init_source.astype(np.complex64).reshape(-1)
    true_source = true_source.astype(np.complex64).reshape(-1)
    true_yf = fft(true_source).reshape(-1)
    true_xf = fftfreq(len(true_source), dt)
    init_yf = fft(init_source).reshape(-1)
    
    idx = np.argsort(true_xf)
    xx = true_xf[idx]
    iyx = (np.abs(init_yf))[idx]
    tyx = (np.abs(true_yf))[idx]
    
    figsize = (6, 6)
    plt.figure(figsize=figsize)
    plt.plot(xx[len(xx)//2:len(xx)//2+200],iyx[len(xx)//2:len(xx)//2+200], label='Initial')
    plt.plot(xx[len(xx)//2:len(xx)//2+200],tyx[len(xx)//2:len(xx)//2+200], label='True')
    plt.legend()
    
    plt.title('source frequency spectra')
    
    plt.savefig(SaveFigPath+'spectra.png')

    plt.close()
    

def PlotSNR(SNR, SaveFigPath):
    """
       Plot SNR between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(SNR[1:],color='purple',lw=1,ls='-',marker='v',markersize=2,label='SNR') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('SNR')
    plt.savefig(str(SaveFigPath)+'SNR.png')
    plt.close()

    
def PlotRSNR(RSNR, SaveFigPath):
    """
       Plot RSNR between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(RSNR[1:],color='green',lw=1,ls='-',marker='v',markersize=2,label='RSNR') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('RSNR')
    plt.savefig(str(SaveFigPath)+'RSNR.png')
    plt.close()

    
def PlotSSIM(SSIM, SaveFigPath):
    """
       Plot SSIM between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(SSIM[1:],color='green',lw=1,ls='-',marker='v',markersize=2,label='SSIM') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('SSIM')
    plt.savefig(str(SaveFigPath)+'SSIM.png')
    plt.close()
    
    
def PlotERROR(ERROR,SaveFigPath):
    """
       Plot ERROR between GT and inverted model
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(6)
    fig1.set_figwidth(6)

    line1, = ax1.plot(ERROR[1:],color='green',lw=1,ls='-',marker='v',markersize=2,label='ERROR') 
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('ERROR')
    plt.savefig(str(SaveFigPath)+'ERROR.png')
    plt.close()
    
