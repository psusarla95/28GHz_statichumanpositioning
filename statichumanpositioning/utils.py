import numpy as np
import pandas as pd 
from scipy import signal, io
#import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cmath

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def load_dataset(data_path, only_highsubj = True, considered_positions=None):
    examples = []
    ref = []
    glob_labels = []
    count = 0

    #for dataset in sorted(data_path.iterdir()):
    #    #if dataset.is_dir():
    #    #    #dataset = dataset / set_pos
    #    #    print(dataset)
    #    #    for sub in sorted(dataset.iterdir()):
    #    #        print('\t', sub.name)
    cs = 0
    for file in sorted(data_path.iterdir()):
        if "sub1_" in file.name or "sub2_" in file.name: # removed because the steering was not operating during the acquisition
            continue
        if only_highsubj:
            if "sub10" in file.name or "sub11" in file.name or "sub15" in file.name or "sub16" in file.name or "sub19" in file.name or "sub22" in file.name:
                continue
        for p in considered_positions:
            if p == 8 and "er" in str(file.name):
                    print('\t\t\t', file.name)
                    ref.append(np.load(file))
                    cs += 1
            if "es" + str(p) in file.name:
                print('\t\t\t', file.name)
                examples.append(np.load(file))
                glob_labels.append(int(str(file)[-5]))
                count += 1
            
    examples.extend(ref)
    glob_labels.extend(list(np.zeros((len(ref))) + 8))
    # labels starting from 1, set labels zero for empty room and fix for the others depending on the considered positions
    labels = np.array(glob_labels) 

    #print(labels)
    if 8 in considered_positions:
        pos_tooccupy = 1
        labels[labels == 8] = 0
    else:
        pos_tooccupy = 0

    while len(considered_positions) > 0:
        min_ = np.min(considered_positions)
        labels[labels == min_] = pos_tooccupy
        pos_tooccupy += 1
        considered_positions = np.setdiff1d(considered_positions,min_)
    
    #print(labels)
 

    print("\n--- TOTAL dataset: num examples:", len(examples), "num labels:", labels.shape[0])
    return examples, labels, ref, glob_labels


def load_dataset_fromscratch(data_path, only_highsubj = True, considered_positions=None, only_amplitude = False, rssi = True, consider_inverse=False, only_inverse = False, freq = None):
    examples_ampl = []
    examples_phase = []
    ref_ampl = []
    ref_phase = []
    if rssi:
        only_amplitude = True # in this case the RSSI use the variables related to amplitude even if it is rssi !!! not really clear, can be improved

    #for dataset in sorted(data_path.iterdir()):
    #    if dataset.is_dir():
    #        #dataset = dataset / set_pos
    #        if os.path.exists(dataset):
    #            #print(dataset)
    #for sub in sorted(data_path.iterdir()):
    #    print('\t', sub.name)
    for file in sorted(data_path.iterdir()):
        #print('\t', file.name)
        if "sub1_" in file.name or "sub2_" in file.name: # removed because the steering was not operating during the acquisition
            continue
        if only_highsubj:
            if "sub10" in file.name or "sub11" in file.name or "sub15" in file.name or "sub16" in file.name or "sub19" in file.name or "sub22" in file.name:
                continue
        if 8 in considered_positions and "er" in str(file.name):
            #print('\t\t\t', file.name)
            ampl, phase = load_extract(file, rssi, freq)
            ref_ampl.append(ampl)
            ref_phase.append(phase)
            
        if "ht" in file.name:
            #print('\t\t\t', file.name)
            ampl, phase = load_extract(file, rssi, freq)
            examples_ampl.append(ampl)

            if not only_amplitude:
                examples_phase.append(phase)
                    
    # SLICING ACOORDING TO DIFFERENT EXHAUSTIVE SEARCHES
    examples_ampl = np.array(examples_ampl)
    #print(examples_ampl.shape)
    ampl_slices, labels = ht_slicing(examples_ampl, considered_positions, rssi=rssi, consider_inverse=consider_inverse, only_inverse = only_inverse)
    #print(ampl_slices.shape)
    
    if not only_amplitude:
        examples_phase = np.array(examples_phase)
        #print(examples_phase.shape)
        phase_slices, labels = ht_slicing(examples_phase, considered_positions, rssi=rssi, consider_inverse=consider_inverse, only_inverse = only_inverse)
        #print(phase_slices.shape)
    
    #print(labels)
    labels = np.array(labels)
    #print(labels)
  
    '''
    fig, axs = plt.subplots(1,3)
    for line in range(ampl_slices.shape[0]):
        #print(ampl_slices[line])
        if line%2==0:
            axs[0].plot(phase_slices[line])
        else:
            axs[1].plot(phase_slices[line])
    '''
    if len(ref_ampl) > 0:

        ref_ampl = np.array(ref_ampl)
        er_ampl_slices  = er_slicing(ref_ampl, rssi=rssi, consider_inverse=consider_inverse, only_inverse = only_inverse)

        if not only_amplitude:
            ref_phase = np.array(ref_phase)
            er_phase_slices  = er_slicing(ref_phase, rssi=rssi, consider_inverse=consider_inverse, only_inverse = only_inverse)
            #print(er_phase_slices.shape)

        '''
        for line in range(er_ampl_slices.shape[0]):
            axs[2].plot(er_phase_slices[line])
        plt.show()
        '''
        ampl_slices = np.concatenate((ampl_slices,er_ampl_slices), axis = 0)

        if not only_amplitude:
            phase_slices = np.concatenate((phase_slices,er_phase_slices), axis = 0)

        er_labels = np.zeros(er_ampl_slices.shape[0])+8
        labels = np.concatenate((labels, er_labels), axis = 0)
        pos_tooccupy = 1
        labels[labels == 8] = 0
    else:
        labels = np.array(labels) 
        pos_tooccupy = 0

    while len(considered_positions) > 0:
        min_ = np.min(considered_positions)
        labels[labels == min_] = pos_tooccupy
        pos_tooccupy += 1
        considered_positions = np.setdiff1d(considered_positions,min_)
    
    #print(labels)
    #print(labels.shape)
    #print(ampl_slices.shape)
    if not only_amplitude:
        print(phase_slices.shape)

    if not rssi:
        ampl_slices = np.reshape(ampl_slices, (ampl_slices.shape[0], ampl_slices.shape[1], ampl_slices.shape[2], 1))
    
    if only_amplitude:
        examples = ampl_slices
    else:
        phase_slices = np.reshape(phase_slices, (phase_slices.shape[0], phase_slices.shape[1], phase_slices.shape[2] , 1))
        examples = np.concatenate((ampl_slices,phase_slices), axis = 3)
    #print("\n--- TOTAL dataset: num examples:", examples.shape, "num labels:", labels.shape[0])
    
    return examples, labels


# All slices are good for er
def er_slicing(data, exh_searches=3, frequencies = 501, rssi = True, consider_inverse = False, only_inverse = False):
    
    start_idxs = np.array((range(exh_searches+1)))*121
    #print(start_idxs)

    if rssi:
        if consider_inverse and not only_inverse:
            slices = np.zeros((exh_searches*data.shape[0], 121, 2))
        else:
            slices = np.zeros((exh_searches*data.shape[0], 121))
         
    else:
        slices = np.zeros((exh_searches*data.shape[0], 121, frequencies))
    
    line = 0
    for data_line in range(data.shape[0]):
        for es in range(exh_searches):
            start = start_idxs[es]
            stop = start_idxs[es+1]
            #print("\n\nSlice:", es, start, stop)
            #print(data[line//exh_searches, start:stop])
            #print(data_line)
            #print(line)
            if consider_inverse:
                new_data_line = np.reshape(invert_data(data[data_line, start:stop]), (data[data_line, start:stop].shape[0],1))
                data_resh = np.reshape(data[data_line, start:stop], (data[data_line, start:stop].shape[0],1))
                if only_inverse:
                    if rssi:
                        slices[line] = np.reshape(new_data_line, new_data_line.shape[0])
                    else:
                        slices[line] = np.reshape(new_data_line, new_data_line.shape[0])
                else:
                    if rssi:
                        slices[line, :, :] = np.concatenate((new_data_line, data_resh), axis = 1)
                    else:
                        slices[line] = np.concatenate((new_data_line, data_resh), axis = 1)

            else:
                if rssi:
                    slices[line] = data[data_line, start:stop]
                else:
                    slices[line] = data[data_line, start:stop, :]
            
            line += 1

    return slices

# Only slices correspondent to considered positions are good
def ht_slicing(data, considered_positions, exh_searches=7, frequencies = 501, rssi = True, consider_inverse = False, only_inverse = False):
    start_idxs = np.array((range(exh_searches+1)))*121
    #print(data.shape)

    considered_positions = np.setdiff1d(considered_positions,8)

    if rssi:
        if consider_inverse and not only_inverse:
            slices = np.zeros((len(considered_positions)*data.shape[0], 121, 2))
        else:
            slices = np.zeros((len(considered_positions)*data.shape[0], 121))
         
    else:
        slices = np.zeros((len(considered_positions)*data.shape[0],121, frequencies))
    line = 0
    labels = []
    
    for data_line in range(data.shape[0]):
        for es in range(exh_searches):
            if es+1 in considered_positions:
                labels.append(es+1)
                start = start_idxs[es]
                stop = start_idxs[es+1]
                #print("\n\nSlice:", es, start, stop)
                #print(data_line)
                #print(line)
                if consider_inverse:
                    new_data_line = np.reshape(invert_data(data[data_line, start:stop]), (data[data_line, start:stop].shape[0],1))
                    data_resh = np.reshape(data[data_line, start:stop], (data[data_line, start:stop].shape[0],1))
                    if only_inverse:
                        if rssi:
                            slices[line] = np.reshape(new_data_line, new_data_line.shape[0])
                        else:
                            slices[line] = np.reshape(new_data_line, new_data_line.shape[0])
                    else:
                        if rssi:
                            slices[line, :, :] = np.concatenate((new_data_line, data_resh), axis = 1)
                        else:
                            slices[line] = np.concatenate((new_data_line, data_resh), axis = 1)
                        
                        
                        
                else:
                    if rssi:
                        slices[line] = data[data_line, start:stop]
                    else:
                        slices[line] = data[data_line, start:stop, :]
            
                line += 1
    #print("labels", labels)
    #print(slices.shape)
    return slices, labels


def load_extract(path, rssi = True, freq=None):
    pna_data = io.loadmat(path)
    pna_val = np.array(pna_data["MeasData"]['S21'][0][0])

    # Select one frequency
    if rssi:
        return 10*np.log10(np.square(np.abs(pna_val[freq,:]))), None
        
    # Compute amplitude
    ampl = np.abs(pna_val)

    # Compute phase
    phase = np.angle(pna_val)

    #plt.plot(ampl, label = "ampl")
    #plt.plot(phase, label = "phase")
    #plt.legend()
    #plt.show()
    return ampl.T, phase.T


def preprocess_data(examples, avg_window = 50, down_factor = 2, n_fft = 30, shift = 10, aug_factor = 1, mode = "diff", coeff_aschannels = False):

    if mode == "time" or mode == "wavelet":
        X = []
    else:
        if coeff_aschannels:
            X = np.zeros([len(examples), n_fft, 2])
        else:
            X = np.zeros([len(examples), n_fft*2])

    for line in range(len(examples)):
        temp = examples[line]
        
        if mode == "time" or mode == "wavelet":
            X.append(temp)        
        else:
            # Interpolate NaN
            condition = temp == -np.inf
            temp[condition] = np.nan
            df_sig = pd.DataFrame(temp)
            df_sig = df_sig.interpolate(method='pchip', limit_direction= 'both', limit_area = None)
        
            if mode == "diff":
                temp = np.array(df_sig).T[0]

                # Moving average
                temp_avg = moving_average(temp, avg_window)

                # Difference with successive sample
                temp_dec = signal.decimate(temp_avg, down_factor)
                temp_succ = temp_dec[1:]
                temp = temp_succ - temp_dec[:-1]
                #print(temp.shape)

            elif mode == "fft":
                temp = np.squeeze(np.array(df_sig))
                #print(temp.shape)

            # Frequency transform
            window = np.hanning(121)
            #print(temp.shape)
            #plt.plot(temp)
            temp = temp * window
            #plt.plot(temp)
            #plt.show()
            #print(temp.shape)
            temp_freq = np.fft.fft(temp,norm='ortho')
            #plt.plot(temp_freq)
            #plt.show()

            if coeff_aschannels:
                X[line,:, 0] = temp_freq.real[temp_freq.real.shape[0]//2]#[shift:n_fft+shift]
                X[line,:, 1] = temp_freq.imag[temp_freq.real.shape[0]//2]#[shift:n_fft+shift]
                Xline = X[line,:,:]
                #X[line,:] = (Xline - np.min(Xline))/(np.max(Xline) - np.min(Xline))
                #X[line,:,:] = (Xline - np.mean(Xline))/np.max(Xline)
            else:
                X[line,:] = np.concatenate((temp_freq.real[shift:n_fft+shift],temp_freq.imag[shift:n_fft+shift]))
                Xline = X[line,:]
                #X[line,:] = (Xline - np.min(Xline))/(np.max(Xline) - np.min(Xline))
                #X[line,:] = (Xline - np.mean(Xline))/np.max(Xline)
    if not mode == "time" and not coeff_aschannels:
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X

def crop_input(sig, input_len): #with some randomness

    sig_len = sig.shape[0]
    if sig_len > input_len:
        diff = sig_len - input_len
        # si può cambiare scegliendo un inizio random, anche multipli per fare data augmentation
        #start = int(diff/2)
        start = int(np.random.rand(1)*diff)
        end = start + input_len
        input_sig = sig[start:end]
    elif sig_len == input_len:
        input_sig = sig
    else:
        diff = input_len - sig_len
        start = int(np.random.rand(1)*diff)
        first_slice = np.zeros((start, 1)) + sig[0]
        last_slice = np.zeros((diff - start,1)) + sig[-1]
        input_sig = np.concatenate((np.concatenate((first_slice, sig), axis = 0), last_slice), axis = 0)
    #input_sig = np.squeeze(input_sig)
    return input_sig


def crop_input2(sig, input_len, down_factor = 1): # aligning the 5th peak

    orig_sig = sig
    if down_factor > 1: 
        sig_dec = signal.decimate(sig.T[0], down_factor)
        sig = np.reshape(sig_dec, (sig_dec.shape[0],1))
    #print(sig.shape)

    shift = 100
    width = 300
    avg_window = 100
    # center the 5th peak
    if down_factor > 1:
        shift = int(shift/down_factor)
        width = int(width/down_factor)
        avg_window = int(avg_window/down_factor)
        input_len = int(input_len/down_factor)
    
    center = int(sig.shape[0]/2) - shift
    avg_sig = moving_average(sig.T[0], avg_window)
    center_values = avg_sig[center-width:center+width]

    peak_idx = center - width + np.argmax(center_values)
    diff = center-peak_idx
    
    if  diff > 0:
        first_slice = np.zeros((diff, 1)) + sig[0]
        input_sig = np.concatenate((first_slice, sig), axis = 0)
    elif diff == 0:
        input_sig = sig
    else:
        last_slice = np.zeros((-diff, 1)) + sig[-1]
        input_sig = np.concatenate((sig, last_slice), axis = 0)

    #plt.plot(sig)
    #plt.plot(orig_sig)
    #plt.plot(avg_sig)
    #plt.plot(input_sig)
    #plt.axvline(x=peak_idx, color='b', label='axvline - full height')
    #plt.axvline(x=center, color='r', label='axvline - full height')
    #plt.show()

    if input_sig.shape[0] > input_len:       
        cut_window = int(input_len/2)

        if center-cut_window<0:
            diff = cut_window-center
            first_slice = np.zeros((diff, 1)) + input_sig[0]
            input_sig = np.concatenate((first_slice, input_sig), axis = 0)
            
            input_sig = input_sig[center+diff-cut_window:center+diff+cut_window]
        else:  
            input_sig = input_sig[center-cut_window:center+cut_window]
    elif input_sig.shape[0] < input_len:
        diff = input_len - input_sig.shape[0]
        pad_window = int(diff/2)
        first_slice = np.zeros((pad_window, 1)) + input_sig[0]
        last_slice = np.zeros((diff-pad_window, 1)) + input_sig[-1]
        input_sig = np.concatenate((np.concatenate((first_slice, input_sig), axis = 0), last_slice), axis = 0)

    return input_sig

def my_model(activation = 'relu', num_classes = 4, input_shape = 60):
    return tf.keras.Sequential([
    tf.keras.layers.Input(input_shape),
    tf.keras.layers.GaussianNoise(0.1, seed = 43),
    tf.keras.layers.Dense(input_shape, activation=None), 
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Activation(activation),
    #tf.keras.layers.GaussianNoise(0.01, seed = 43),
    #tf.keras.layers.Dense(input_shape//2, activation=None), 
    #tf.keras.layers.BatchNormalization(axis=-1),
    #tf.keras.layers.Activation(activation),
    
    #tf.keras.layers.Dropout(0.1, seed = 43),
    #tf.keras.layers.Dense(400, activation=None), 
    #tf.keras.layers.BatchNormalization(axis=-1),
    #tf.keras.layers.Activation(activation),
    #tf.keras.layers.Dense(40, activation=None), 
    #tf.keras.layers.BatchNormalization(axis=-1),
    #tf.keras.layers.Activation(activation),
    #tf.keras.layers.Dropout(0.1, seed = 43),
    tf.keras.layers.Dense(num_classes, activation=None)
    ])

def my_model_conv(activation = 'relu', conv_feature_maps=4,conv_kernel=6, conv_strides=2, conv_padding='same',
                  pool_size=2, pool_strides=2, pool_padding='same', num_classes = 4, input_shape = 60):
    
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(conv_feature_maps, conv_kernel, strides=conv_strides, padding=conv_padding, activation=None),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.MaxPooling1D(pool_size, pool_strides, padding=pool_padding),
        #tf.keras.layers.Conv1D(conv_feature_maps, conv_kernel, strides=conv_strides, padding=conv_padding, activation=None),
        #tf.keras.layers.BatchNormalization(axis=-1),
        #tf.keras.layers.Activation(activation),
        #tf.keras.layers.MaxPooling1D(pool_size, pool_strides, padding=pool_padding),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4, seed = 43),
        tf.keras.layers.Dense(num_classes, activation=None)
    ])   

def my_model_conv2d(activation = 'relu', conv_feature_maps=4,conv_kernel=6, conv_strides=2, conv_padding='same',
                  pool_size=2, pool_strides=2, pool_padding='same', num_classes = 4, input_shape = 60):
    
    return tf.keras.Sequential([
        tf.keras.layers.GaussianNoise(0.1, seed = 43),
        tf.keras.layers.Conv2D(conv_feature_maps, conv_kernel, strides=conv_strides, padding=conv_padding, activation=None),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.MaxPooling2D(pool_size, pool_strides, padding=pool_padding),
        #tf.keras.layers.Conv2D(int(conv_feature_maps/2), int(conv_kernel/2), strides=int(conv_strides/2), padding=conv_padding, activation=None),
        #tf.keras.layers.BatchNormalization(axis=-1),
        #tf.keras.layers.Activation(activation),
        #tf.keras.layers.MaxPooling2D(int(pool_size/2), int(pool_strides/2), padding=pool_padding),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.4, seed = 43),
        tf.keras.layers.Dense(num_classes, activation=None),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Activation(activation),
        tf.keras.layers.Dense(num_classes, activation=None)
    ])

# To artificially invert the roles of RX and TX
# in the experiment RX was fixed for the 11th positions of TX
# here we exchange the role
# so if before it was 0 1 2 3 4 5 6 7 8 9 10 - 11 12 13 14 ...
# now it will be 0 11 22 33 44 55 66 77 88 99 110 - 1 12 23 ...  
def invert_data(data_line, angles = 11): 
    start_idxs = np.array((range(angles+1)))*angles
    new_data_line = np.zeros(data_line.shape)
    #print(start_idxs)
    for es in range(angles):
        new_data_line[start_idxs[es]:start_idxs[es+1]] = data_line[start_idxs[:-1]+es]
    #plt.plot(np.concatenate((data_line, new_data_line), axis = 0))
    #plt.show()
    return new_data_line
