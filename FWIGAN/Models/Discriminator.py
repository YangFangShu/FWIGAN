"""

@author: fangshuyang (yangfs@hit.edu.cn)
Build the Discriminator

"""

import torch.nn as nn
from torch.autograd import grad
import torch
import torch.nn.init as init
 
######################################

## Discriminator (6ConvBlock(Conv2d+LeakyReLU+MaxPool)+2FullyConnectedLayer)

######################################
# Initialization of network parameters
def weights_init(m, leak_value):
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.weight is not None:

            init.kaiming_normal_(m.weight, a = leak_value)

        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.kaiming_normal_(m.weight, a = leak_value)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    
class Discriminator(nn.Module):
    def __init__(self,batch_size,ImagDim, LReLuRatio,filters,leak_value):
        super(Discriminator, self).__init__()

        self.truth_channels = 1
        self.batch_size = batch_size
        self.filters = filters
        self.LReLuRatio = LReLuRatio
        self.ImagDim = ImagDim
        self.leak_value = leak_value
        
        
        self.conv1 = nn.Conv2d(self.truth_channels, self.filters[0], kernel_size=3, stride=1, padding=1)
        self.ac1 = nn.LeakyReLU(self.LReLuRatio)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(self.filters[0], self.filters[1], kernel_size=3, stride=1, padding=1)
        self.ac2 = nn.LeakyReLU(self.LReLuRatio)
        self.pool2 = nn.MaxPool2d(2,2) 
        self.conv3 = nn.Conv2d(self.filters[1], self.filters[2], kernel_size=3, stride=1, padding=1)
        self.ac3 = nn.LeakyReLU(self.LReLuRatio)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(self.filters[2], self.filters[3], kernel_size=3, stride=1, padding=1)
        self.ac4 = nn.LeakyReLU(self.LReLuRatio)
        self.pool4 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(self.filters[3], self.filters[4], kernel_size=3, stride=1, padding=1)
        self.ac5 = nn.LeakyReLU(self.LReLuRatio)
        self.pool5 = nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(self.filters[4], self.filters[5], kernel_size=3, stride=1, padding=1)
        self.ac6 = nn.LeakyReLU(self.LReLuRatio)
        self.pool6 = nn.MaxPool2d(2,2)
        ## here need to be changed based on the size of amplitudes
        # for mar_big: [2000,567]==>[31*8] (fc1:1000); mar_smal:[2000,310]==>[31*4](fc1:2000); over:[2000,400]==>[31,6](fc1:1500)
        self.fc1 = nn.Linear(31*8*filters[5], 2000) 
        self.ac7 = nn.LeakyReLU(self.LReLuRatio)        
        self.fc2 = nn.Linear(1500, 1)
        
        
    def forward(self, input):
        # input shape: [num_shots_per_batch,1,nt,num_receiver_per_shot]
        output = input.reshape(self.batch_size,self.truth_channels,self.ImagDim[0],self.ImagDim[1])
        output = self.ac1(self.pool1(self.conv1(output)))
        output = self.ac2(self.pool2(self.conv2(output)))
        output = self.ac3(self.pool3(self.conv3(output)))
        output = self.ac4(self.pool4(self.conv4(output)))
        output = self.ac5(self.pool5(self.conv5(output)))
        output = self.ac6(self.pool6(self.conv6(output)))
        output = output.view(-1,31*8*1024) # here the last dim should be same as fc1
        output = self.fc1(output)
        output = self.ac7(output)
        output = self.fc2(output) 
        output = output.view(-1)
        return output 

                          
    
                          
  