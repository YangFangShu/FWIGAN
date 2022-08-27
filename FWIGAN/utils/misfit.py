"""
Common functions for calculating various loss

@author: fangshuyang (yangfs@hit.edu.cn)


"""
import numpy as np
from scipy.signal import butter, lfilter,hanning
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
import math
from math import exp
import scipy.stats


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, channel, lamb, device):
    """
        Gradient penalty term for training Discriminator (refer to the WGAN_GP)
    """
    if batch_size != real_data.shape[0] or channel != real_data.shape[1]:
        assert False, "The batch size or channel is wrong!!!"
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    dim = real_data.size()
    alpha = alpha.view(batch_size, channel, dim[2], dim[3])
    alpha = alpha.float().to(device)
    
    fake_data = fake_data.view(batch_size, channel, dim[2], dim[3])
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                          grad_outputs=torch.ones_like(disc_interpolates).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty





def transform(f, g, trans_type, theta):
    """
        do the transform for the signal f and g
        Args:
            f, g: seismic data with the shape [num_time_steps,num_shots*num_receivers_per_shot]
            trans_type: type for transform
            theta:the scalar variable for transform            
        return:
            output with transfomation
    """ 
    assert len(f.shape) == 2
    c = 0.0 
    device = f.device
    if trans_type == 'linear':
        min_value = torch.min(f.detach().min(), g.detach().min())
        mu, nu = f, g
        c = -min_value if min_value<0 else 0
        c = c * theta
        d = torch.ones(f.shape).to(device)
    elif trans_type == 'abs':
        mu, nu = torch.abs(f), torch.abs(g)
        d = torch.sign(f).to(device)
    elif trans_type == 'square':
        mu = f * f
        nu = g * g
        d = 2 * f
    elif trans_type == 'exp':
        mu = torch.exp(theta*f)
        nu = torch.exp(theta*g)
        d = theta * mu
    elif trans_type == 'softplus':
        mu = torch.log(torch.exp(theta*f) + 1)
        nu = torch.log(torch.exp(theta*g) + 1)
        d = theta / torch.exp(-theta*f)
    else:
        mu, nu = f, g
        d = torch.ones(f.shape).to(device)
    mu = mu + c + 1e-18 
    nu = nu + c + 1e-18
    return mu, nu, d

def trace_sum_normalize(x):
    """
    normalization with the summation of each trace
    note that the channel should be 1
    """
    x = x / (x.sum(dim=0,keepdim=True)+1e-18)
    return x

def trace_max_normalize(x):
    """
    normalization with the maximum value of each trace (the value of each trace is in [-1,1] after the processing)
    note that the channel should be 1
    """
    x_max,_ = torch.max(x.abs(),dim=0,keepdim=True)
    x = x / (x_max+1e-18)
    return x

def shot_max_normalize(x):
    """
    normalization with the maximum value of each shot (the value of each shot is in [-1,1] after the processing)
    note that the channel should be 1
    """
    num_shots, channel, num_time_steps, num_receivers = x.shape
    x_max,_ = torch.max(x.detach().reshape(num_shots,channel,num_time_steps*num_receivers).abs(),dim=2,keepdim=True)
    x = x / (x_max.repeat(1,1,num_time_steps*num_receivers).reshape(num_shots,channel,num_time_steps,num_receivers))
    return x



class NIMFunction(Function):
    """
    Normalized Integration Method, Liu et al., 2012: the objective function measures the misfit between the integral of the absolute value, or of the square, or of the envelope of the signal.
    F_i = \frac{\sum_{j=1}^i P(f_j)}{\sum_{j=1}^n P(f_j)}, 

    G_i = \frac{\sum_{j=1}^i P(g_j)}{\sum_{j=1}^n P(g_j)}, 

    \ell(f, g) = \sum_{i=1}^n |F_i - G_i|^p, 

    where function :`P` is choosed to make the vector nonnegative, 
    e.g. :`|x|`, `|x|^2`.

    Args:
    p (real): the norm degree. Default: 2 
    trans_type : the nonnegative transform. Default: 'linear'
    theta : the parameter used in nonnegtive transform. Default: 1
    Note:
    NIM is equivalent to Wasserstein-1 distance (Earth Mover's distance) when 
    p = 1
    """
    @staticmethod
    def forward(ctx, f, g, p, trans_type='linear', theta=1.):
        assert p >= 1
        
        assert f.shape == g.shape
        assert len(f.shape) == 2
        device = f.device
        p = torch.tensor(p).to(device)
        num_time_steps,num_trace_total = f.shape

        mu, nu, d = transform(f, g, trans_type, theta)
        
        mu = trace_sum_normalize(mu)
        nu = trace_sum_normalize(nu)
        
        F = torch.cumsum(mu, dim=0)
        G = torch.cumsum(nu, dim=0)
        
        #result = (torch.abs(F - G) ** p).sum()
        ctx.save_for_backward(F-G, mu, p, d)
        
        return (torch.abs(F - G) ** p).sum()

    @staticmethod
    def backward(ctx, grad_output):
        residual, mu, p, d = ctx.saved_tensors
        
        if p == 1:
            df = torch.sign(residual) * mu *d
        else:
            df = (residual) ** (p - 1) * mu * d     
        return df, None, None, None, None
         

def NIM(f, g, p):
    return NIMFunction.apply(f, g, p)



def Wasserstein1(f,g,trans_type,theta):
    assert f.shape == g.shape
    assert len(f.shape) == 3
    device = f.device
    p = 1
    num_time_steps,num_shots_per_batch,num_receivers_per_shot = f.shape
    f = f.reshape(num_time_steps,num_shots_per_batch*num_receivers_per_shot)
    g = g.reshape(num_time_steps,num_shots_per_batch*num_receivers_per_shot)

    mu, nu, d = transform(f, g, trans_type, theta)
    
    assert mu.min() > 0
    assert nu.min() > 0
    
    mu = trace_sum_normalize(mu)
    nu = trace_sum_normalize(nu)

    F = torch.cumsum(mu, dim=0)
    G = torch.cumsum(nu, dim=0)
    
    w1loss = (torch.abs(F - G) ** p).sum()
    return w1loss

