"""
  
    @author: fangshuyang (yangfs@hit.edu.cn)
    
    Forward Simulation of 2D Acoustic Wave Equation 
    (4- or 8- order finite-difference scheme in space domain and 
    2-order finite-difference scheme in time domain. 
    The boundary absorbing condition is PML condition. Free surface can be implemented.)


"""
import torch
import deepwave
import numpy as np
import torch.nn as nn
from utils.utils import AddAWGN

"""
   Define the Physics Simulator Module (e.g., forward wave propagation of AWE)
"""
class PhySimulator(nn.Module):
    def __init__(self, dx,num_shots,num_batches,x_s,x_r,dt,pml_width,order,survey_pad):
        super(PhySimulator,self).__init__()
        self.dx = dx
        self.num_shots = num_shots
        self.num_batches = num_batches
        self.x_s = x_s
        self.x_r = x_r
        self.dt = dt
        self.num_shots_per_batch = int(self.num_shots / self.num_batches)
        self.pml_width = pml_width
        self.order = order
        self.survey_pad = survey_pad
        
        
    def forward(self,model,source_amplitudes,it,criticIter,j,status, \
                AddNoise,noi_var,learnAWGN):
       
        prop = deepwave.scalar.Propagator({'vp': model}, self.dx,self.pml_width, self.order,self.survey_pad)
        batch_src_amps = source_amplitudes.repeat(1, self.num_shots_per_batch, 1)
        if status == 'TD':
            
            # for the inner loop of training Critic
            if it*criticIter+j < self.num_batches:
                batch_x_s = self.x_s[it*criticIter+j::self.num_batches]
                batch_x_r = self.x_r[it*criticIter+j::self.num_batches]
            else:
                batch_x_s = self.x_s[((it*criticIter+j) % self.num_batches)::self.num_batches]
                batch_x_r = self.x_r[((it*criticIter+j) % self.num_batches)::self.num_batches]

        elif status == 'TG':
            batch_x_s = self.x_s[it::self.num_batches]
            batch_x_r = self.x_r[it::self.num_batches]
        else:
            assert False, 'Please check the status of training!!!'

        batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, self.dt)
        
        if AddNoise == True and noi_var != None and learnAWGN == True:
            batch_rcv_amps_pred =  AddAWGN(batch_rcv_amps_pred, noi_var)
            
       
        return batch_rcv_amps_pred