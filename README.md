# FWIGAN
<a href="https://zenodo.org/badge/latestdoi/529577443"><img src="https://zenodo.org/badge/529577443.svg" alt="DOI"></a>

This repo contains a PyTorch implementation with DeepWave for the paper [FWIGAN: Full-waveform inversion via a physics-informed generative adversarial network](), which is submitted to the Journal of Geophysical Research-Solid Earth. The preliminary arxiv version of this project (Revisit Geophysical Imaging in A New View of Physics-informed Generative Adversarial Learning) is availabel [here](https://arxiv.org/abs/2109.11452).

![Flowchart of FWIGAN](/images/flowchat.png)

## Abstract
Full-waveform inversion (FWI) is a powerful geophysical imaging technique that produces high-resolution subsurface physical parameters by iteratively minimizing the misfit between simulated and observed seismograms. Unfortunately, conventional FWI with least-squares loss function suffers from many drawbacks such as the local-minima problem and human intervention in the fine-tuning of parameters. It is particular troubling with contaminated measurements and unexpected starting models. Recent works that rely on partial differential equations and neural networks show promising performances for two-dimensional FWI. Inspired by the competitive learning of generative adversarial networks, we proposed an unsupervised learning paradigm that integrates the wave equation with a discriminative network to accurately estimate physically consistent velocity models in a distribution sense (FWIGAN). The introduced framework does not need a labelled training dataset or pretraining of the network; therefore,  this framework is flexible and able to achieve inversion with minimal user interaction. We experimentally validated our method on three baseline geological models, and the comparable results demonstrate that FWIGAN faithfully recovers the velocity models and outperforms other traditional or deep learning-based algorithms. Furthermore, benefit from the physics-constrained learning in this method, FWIGAN paves the way to sidestep the local-minima issue by reducing the sensitivity to initial models or data noise.

## Prerequisites
```
python 3.8.13  
pytorch 1.7.1
scipy 1.8.0
numpy 
matplotlib 3.5.1
scikit-image 0.19.2
math
jupyter
pytest 6.2.4
IPython
deepwave 0.0.8
```
#### NOTE 
We install the deepwave with https://github.com/guoketing/deepwave-order.git which provides finite difference scheme with 8th-order accuracy in the space domain. ```cd ./deepwave-order/``` and run ```python setup.py install ``` to achieve the installation. Moreover, it is better to use Conda for the installation of all dependecies.

## Run the code
Enter the FWIGAN folder
```
cd ./FWIGAN/
```
Correct the parameter settings and data path in ```ParamConfig.py``` and ```PathConfig.py```

e.g.
```
####################################################
####   MAIN PARAMETERS FOR FORWARD MODELING         ####
####################################################

peak_freq = 5.                 # central frequency
peak_source_time = 1 / peak_freq  # the time (in secs) of the peak amplitude
dx        = 30.0               # step interval along x/z direction
dt        = 0.003              # time interval (e.g., 3ms)
num_dims  = 2                  # dimension of velocity model
nz        = 100                # model shape of z dimension (depth)
ny        = 310                # model shape of y dimension (when num_dims>=2)
```
and 
```
####################################################
####                   FILENAMES               ####
####################################################
# Define the data （'mar_smal','mar_big','over'）
dataname  = 'mar_smal'
sizestr   = '_100_310'
```
Then, run the following script to generate dataset and implement traditional FWI 
```
python FWI_main.py
```
The initla model, source amplitude, and observed data are saved in the result path.

Next, run the following script to implement FWIGAN
```
python FWIGAN_main.py
```
For the implementation of FWIDIP, run the follwing script to pretrain the network
```
python dip_pretrain.py
```
After the pretraining is done, you can run the following script to implement FWIDIP
```
python FWIDIP.py
```

## Results
Outputs can be found in ```/FWIGAN/results/...```.
### Visual examples
#### 1. Marmousi Model:
![Inversion results of the Marmousi2 model.](/images/mar_smal_rec.png)

#### 2. Marmousi2 Model:
![Inversion results of the Marmousi2 model.](/images/mar_big_rec.png)

#### 3. Overthrust Model:
![Inversion results of the Marmousi2 model.](/images/over_rec.png)

## Citation

If you find the paper and the code useful in your research, please cite the paper:
```
@article{yang2021revisit,
  title={Revisit Geophysical Imaging in A New View of Physics-informed Generative Adversarial Learning},
  author={Yang, Fangshu and Ma, Jianwei},
  journal={arXiv preprint arXiv:2109.11452},
  year={2021}
}
```
If you have any questions about this work, feel free to contract us: yangfs@hit.edu.cn
