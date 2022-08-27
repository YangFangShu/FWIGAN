"""
Spyder Editor


@author: fangshuyang (yangfs@hit.edu.cn)

"""

# Need to restart runtime before this step
from __future__ import print_function
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deepwave
import os
from timeit import default_timer as timer
import scipy.io as spio
import math

from utils.utils import *
from utils.plotting import *
from utils.misfit import *
from utils.PhySimulator import PhySimulator
from torch import optim
from IPython.core.debugger import set_trace 
from Models.Discriminator import Discriminator
from Models.Discriminator import weights_init
from Models.skip import skip,optimize
