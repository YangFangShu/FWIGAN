##Record the path

"""
Spyder Editor

@author: fangshuyang (yangfs@hit.edu.cn)

"""

from ParamConfig import *
import os
####################################################
####                   FILENAMES               ####
####################################################
# Define the data （'mar_smal','mar_big','over'）
dataname  = 'mar_smal'
sizestr   = '_100_310'

####################################################
####                  PATHS              
####################################################

#### Main path
main_dir = '/home/fangshu/Code/pytorch/FWIGAN/'
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!!')

# Result path 
if not os.path.exists('./results/'):
    os.makedirs('./results/') 
    
results_dir      = main_dir + 'results/'

    
ResultPath = results_dir + str(dataname)+'_dim'+str(vmodel_dim)+'_pf'+str(peak_freq)+'_dx'+str(dx)+'_dt'+ \
str(dt)+'_T'+str(total_t)+'_ns'+str(num_shots)+'_nrps'+str(num_receivers_per_shot)+ \
'_sig'+str(gfsigma)


if lipar != None:
    ResultPath = ResultPath+'_lip'+str(lipar)
    
        
if source_depth > 0:
    ResultPath = ResultPath+'_sd'+str(source_depth)

    
if receiver_depth > 0:
    ResultPath = ResultPath+'_rd'+str(receiver_depth)
   
    
if fix_value_depth > 0:
    ResultPath = ResultPath+'_fixv'+str(fix_value_depth)
    
    
if order != 4:
    ResultPath = ResultPath+'_order'+str(order)
   
    
if pml_width != 10 and pml_width != None:
    ResultPath = ResultPath+'_pml'+str(pml_width)
       
    
ResultPath = ResultPath + '/'
  
    
if not os.path.exists(ResultPath):    
    os.makedirs(ResultPath)    
    

    
    
#  Data path
datapath = '/home/fangshu/Code/pytorch/FWIGAN/'
if not os.path.exists('./data/'):
    os.makedirs('./data/')   

data_dir      = datapath + 'data/'   
data_path = data_dir+str(dataname)+str(sizestr)+'.bin'