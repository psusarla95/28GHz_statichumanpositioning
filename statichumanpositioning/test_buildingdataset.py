import numpy as np
np.random.seed(43)
#import tensorflow as tf
#tf.random.set_seed(43)
import random
random.seed(43)

import pathlib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
#from keras.utils import to_categorical
import os
from datetime import datetime
#import tensorflow_addons as tfa 
from utils import *
import pywt
import matplotlib

########################################################################################################################################
# PART-1: SET THE HYPER PARAMETERS AND FLAGS DEPENDING ON THE ANALYSIS
#########################################################################################################################################
#print("FREQ INDEX ",freq)
net_type = "ffnn" # OPTIONS: "ffnn", "conv1D", "conv1Dchannels", "conv2D"
only_highsubj =  False # True to have only subjects higher than 160 cm
mode = "freq" # OPTIONS: "time", "fft", "diff", "wavelet"
considered_positions = [1,2,3,4,5,6,7,8] # 8 is empty room
consider_inverse = False
only_inverse = False
only_rssi = False #CSI info is needed then set only_rssi = False

##########################################################################################################################################
# PART-2: TUNED HYPER PARAMETER AND FLAGS 
##########################################################################################################################################
freq = 1
aug_factor = 1
n_fft = 30
down_factor_time = 1
scales = range(1, 50)
waveletname = 'haar' # mexh, morl, gaus1, shan -  haar, bior family


if net_type == "ffnn":
    conv = False #to use convolutional models
    coeff_aschannels = False
    conv2D = False
elif net_type == "conv1D":
    conv = True
    coeff_aschannels = False  # to use only when conv is True: to consider real and imaginary part as different channels
    conv2D = False
elif net_type == "conv1Dchannels":
    conv = True
    coeff_aschannels = True
    conv2D = False
elif net_type == "conv2D":
    conv2D = True
    conv = False
    coeff_aschannels = False

if mode == "time": 
    input_len = 121
elif mode == "wavelet":
    input_len = scales[-1]
elif coeff_aschannels or conv2D:
    input_len = n_fft
else:
    input_len = 2*n_fft


if only_highsubj:
    dataset_type = "freq"  + str(freq) + '_' + '-' + mode + '_' + str(sorted(considered_positions)) + '_' + net_type + '_only_highsubj'
else:
    dataset_type = "freq"  + str(freq) + '_' + '-' + mode + '_' + str(sorted(considered_positions)) + '_' + net_type + '_alsoshortsubj'
    
if consider_inverse and not only_inverse: 
    dataset_type = dataset_type + "_withinverse"
elif consider_inverse and only_inverse: 
    dataset_type = dataset_type + "_onlyinverse"

data_path = pathlib.Path('dataset/pna_data/')
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = dataset_type #current_time + '_' + dataset_type
os.makedirs(folder_path, exist_ok=True)

#**************************************************************************************************************************************************
# PART-3: LOAD THE DATA, PREPROCESS and BUILD CORRESPONDING DATASET
#***************************************************************************************************************************************************
examples, labels = load_dataset_fromscratch(data_path=data_path, 
                                            only_highsubj = only_highsubj, 
                                            considered_positions=considered_positions, 
                                            rssi = only_rssi, 
                                            consider_inverse=consider_inverse,
                                            only_inverse=only_inverse,
                                            freq = freq)
print(examples.shape)
X = preprocess_data(examples, avg_window = 1, down_factor = 1,
                    n_fft = n_fft, shift = 0, mode = mode,
                    coeff_aschannels=coeff_aschannels)

print(X.shape)
print(labels.shape)
