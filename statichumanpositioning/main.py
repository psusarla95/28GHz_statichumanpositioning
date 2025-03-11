import numpy as np
np.random.seed(43)
import tensorflow as tf
tf.random.set_seed(43)
import random
random.seed(43)

import pathlib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from keras.utils import to_categorical
import os
from datetime import datetime
import tensorflow_addons as tfa 
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
                                            rssi = True, 
                                            consider_inverse=consider_inverse,
                                            only_inverse=only_inverse,
                                            freq = freq)
#print(examples.shape)
X = preprocess_data(examples, avg_window = 1, down_factor = 1,
                    n_fft = n_fft, shift = 0, mode = mode,
                    coeff_aschannels=coeff_aschannels)

#**************************************************************************************************************************************************
# PART-4: SELECT Training parameters, models and PERFORM TRAIN-TEST SPLIT over pre-processed dataset
#***************************************************************************************************************************************************
epochs = 300
batch_size = 16
activation='relu'  
num_classes = int(np.max(np.unique(labels)) + 1)

# Models to validate
if mode == "time":
    input_shape = input_len
elif mode == "wavelet":
    input_shape = scales[-1]
else:
    input_shape = X.shape[1]


if conv:
    models = [my_model_conv(num_classes=num_classes, input_shape=input_shape), 
                my_model_conv(num_classes=num_classes, input_shape=input_shape), 
                my_model_conv(num_classes=num_classes, input_shape=input_shape), 
                my_model_conv(num_classes=num_classes, input_shape=input_shape),
                my_model_conv(num_classes=num_classes, input_shape=input_shape), 
                my_model_conv(num_classes=num_classes, input_shape=input_shape)]
elif conv2D:
    models = [my_model_conv2d(num_classes=num_classes, input_shape=input_shape), 
            my_model_conv2d(num_classes=num_classes, input_shape=input_shape), 
            my_model_conv2d(num_classes=num_classes, input_shape=input_shape), 
            my_model_conv2d(num_classes=num_classes, input_shape=input_shape),
            my_model_conv2d(num_classes=num_classes, input_shape=input_shape), 
            my_model_conv2d(num_classes=num_classes, input_shape=input_shape)]
else:
    models = [ my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape),
               my_model(num_classes=num_classes, input_shape=input_shape)]


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, shuffle=True, stratify=labels, random_state=43)

#******************************************************************************************************************************************************************************
# PART-5: PERFORM CROSS VALIDATION USING THE (X_train, y_train) training and (X_test, y_test) testing datasets
#*******************************************************************************************************************************************************************************
n_splits = len(models)
k_fold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=43)
s = 0
train_metrics = {"accuracy":[], "precision":[], "recall":[], "loss":[]}
val_metrics = {"val_accuracy":[], "val_precision":[], "val_recall":[], "val_loss":[]}
cf_mat = []

for train_indices, val_indices in k_fold.split(X_train, y_train):
    if mode == "wavelet":
        channels = scales[-1]
    elif coeff_aschannels or conv2D:
        channels = 2
    else:
        channels = 1
    
    X_train_fold = np.zeros([len(train_indices)*aug_factor, input_len, channels])
    y_train_fold = np.zeros([len(train_indices)*aug_factor])
    
    X_val_fold = np.zeros([len(val_indices), input_len, channels])
    y_val_fold = np.zeros([len(val_indices)])

    # Augment only for train dataset
    abs_line = 0
    for line in range(len(train_indices)):
        temp = np.array(X_train[train_indices[line]]) 
        #print(temp.shape)
        #plt.plot(temp)
        #plt.show()
        for ag in range(aug_factor):
            #X_train_fold[abs_line,:] = crop_input2(temp, input_len, down_factor_time)
            if mode == "time":
                if coeff_aschannels or conv2D:
                    temp = np.reshape(temp, (temp.shape[0],2))
                else:
                    temp = np.reshape(temp, (temp.shape[0],1))

            if mode == "fft" or mode == "diff":
                if coeff_aschannels or conv2D:
                    temp = np.reshape(temp, (n_fft,2))
                else:
                    temp = np.reshape(temp, (temp.shape[0],1))

            if mode == "wavelet":
                sig = np.reshape(temp, (temp.shape[0],))
                coeff, freq = pywt.cwt(sig, scales = scales, wavelet = waveletname, sampling_period = 0.001)
                coeff_ = coeff[:,:len(scales)]   
                temp = coeff_ #np.reshape(coeff_, (coeff_.shape[0], coeff_.shape[1],1))
            
            if ag > 1:
                noise = np.random.normal(0,0.01,temp.shape)
                X_train_fold[abs_line,:] = temp + noise
            else:
                X_train_fold[abs_line,:] = temp
            y_train_fold[abs_line] = y_train[train_indices[line]]

            Xline = X_train_fold[abs_line,:]

            # Normalization before training - in case you add remember to add also for test data
            #X_train_fold[abs_line,:] = (Xline - np.min(Xline))/(np.max(Xline) - np.min(Xline))
            #X_train[abs_line,:] = (Xline - np.mean(Xline))/(np.max(Xline))
            #X_train[abs_line,:] = (Xline - np.mean(Xline))/(np.max(np.abs(Xline)))
            
            abs_line += 1

    for line in range(len(val_indices)):   
        temp = np.array(X_train[val_indices[line]])
        #X_val_fold[line,:] = crop_input2(temp, input_len, down_factor_time)
        if mode == "time":
            if coeff_aschannels or conv2D:
                temp = np.reshape(temp, (temp.shape[0],2))
            else:
                temp = np.reshape(temp, (temp.shape[0],1))

        if mode == "fft" or mode == "diff":
            if coeff_aschannels or conv2D:
                temp = np.reshape(temp, (n_fft,2))
            else:
                temp = np.reshape(temp, (temp.shape[0],1))

        if mode == "wavelet":
            sig = np.reshape(temp, (temp.shape[0],))
            coeff, freq = pywt.cwt(sig, scales = scales, wavelet = waveletname, sampling_period = 0.001)
            coeff_ = coeff[:,:len(scales)]   
            temp = coeff_ # np.reshape(coeff_, (coeff_.shape[0], coeff_.shape[1],1))
        
        X_val_fold[line,:] = temp
        y_val_fold[line] = y_train[val_indices[line]]

        Xline = X_val_fold[line,:]

        # Normalization before training - in case you add remember to add also for test data
        #X_val_fold[line,:] = (Xline - np.mean(Xline))/(np.max(Xline) - np.min(Xline))
        #X_val[line,:] = (Xline - np.mean(Xline))/(np.max(np.abs(Xline)))
        #X_val[line,:] = (Xline - np.mean(Xline))/(np.max(Xline))
            

    if conv and not coeff_aschannels:
        X_train_fold = np.reshape(X_train_fold, (X_train_fold.shape[0], X_train_fold.shape[1], 1))
        X_val_fold = np.reshape(X_val_fold, (X_val_fold.shape[0], X_val_fold.shape[1], 1))
    
    if conv2D:
        X_train_fold = np.reshape(X_train_fold, (X_train_fold.shape[0], X_train_fold.shape[1], X_train_fold.shape[2], 1))
        X_val_fold = np.reshape(X_val_fold, (X_val_fold.shape[0], X_val_fold.shape[1], X_val_fold.shape[2], 1))
    

    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
    # Train data plot
    if coeff_aschannels or conv2D:
        fig, axs = plt.subplots(8,2)
    else:
        fig, axs = plt.subplots(4,2, sharex=True, sharey=True, figsize = (10,10))
    plt.tight_layout()
    plt.subplots_adjust(left = 0.1, right = 0.99, top = 0.95, bottom=0.1, wspace=0.05,hspace=0.2)
    for line in range(X_train_fold.shape[0]):
        if coeff_aschannels or conv2D:
            idx_y = (int(y_train_fold[line]))%2
            idx_x = (int(y_train_fold[line]))//2

            axs[idx_x, idx_y].plot(X_train_fold[line,:,0])
            axs[idx_x+2, idx_y].plot(X_train_fold[line,:,1])
        else:
            idx_y = (int(y_train_fold[line]))%2
            idx_x = (int(y_train_fold[line]))//2

            axs[idx_x, idx_y].plot(X_train_fold[line,:], linewidth=0.5)

    axs[3, 0].set_xlabel("Samples", fontsize = 18)
    axs[3, 1].set_xlabel("Samples", fontsize = 18)

    axs[0, 0].set_ylabel("RSSI (dBm)", fontsize = 18)
    axs[1, 0].set_ylabel("RSSI (dBm)", fontsize = 18)
    axs[2, 0].set_ylabel("RSSI (dBm)", fontsize = 18)
    axs[3, 0].set_ylabel("RSSI (dBm)", fontsize = 18)
    
    cons_pos = considered_positions.copy()
    if 8 in cons_pos:
        axs[0,0].set_title("Empty room", fontsize = 20)
        pos_tooccupy = 1
        cons_pos = np.setdiff1d(cons_pos,8)
    else: 
        pos_tooccupy = 0

    while len(cons_pos) > 0:
        min_ = np.min(cons_pos)
        idx_x = pos_tooccupy%2
        idx_y = pos_tooccupy//2 
        axs[idx_y,idx_x].set_title("Position " + str(min_),  fontsize = 20)
        pos_tooccupy += 1
        cons_pos = np.setdiff1d(cons_pos,min_)


    plt.savefig(folder_path + '/train_data_Kfold_' + str(s) + '.pdf', dpi = 500)
    #plt.show()
    plt.close()

  
    # Val data plot
    if coeff_aschannels or conv2D:
        fig, axs = plt.subplots(8,2)
    else:
        fig, axs = plt.subplots(4,2)

    for line in range(X_val_fold.shape[0]):
        if coeff_aschannels or conv2D:
            idx_y = (int(y_val_fold[line]))%2
            idx_x = (int(y_val_fold[line]))//2

            axs[idx_x, idx_y].plot(X_val_fold[line,:,0])
            axs[idx_x+2, idx_y].plot(X_val_fold[line,:,1])
        else:
            idx_y = (int(y_val_fold[line]))%2
            idx_x = (int(y_val_fold[line]))//2

            axs[idx_x, idx_y].plot(X_val_fold[line,:])
    

    cons_pos = considered_positions.copy()
    if 8 in cons_pos:
        axs[0,0].set_title("Empty room")
        pos_tooccupy = 1
        cons_pos = np.setdiff1d(cons_pos,8)
    else: 
        pos_tooccupy = 0

    while len(cons_pos) > 0:
        min_ = np.min(cons_pos)
        idx_x = pos_tooccupy%2
        idx_y = pos_tooccupy//2 
        axs[idx_y,idx_x].set_title("Position " + str(min_))
        pos_tooccupy += 1
        cons_pos = np.setdiff1d(cons_pos,min_)
    plt.savefig(folder_path + '/val_data_Kfold' + str(s) + '.jpg', dpi = 500)
    #plt.show()
    plt.close()

    #y_train_oh = tf.one_hot(y_train_fold, num_classes)
    #y_val_oh = tf.one_hot(y_val_fold, num_classes)
    y_train_oh = to_categorical(y_train_fold)
    y_val_oh = to_categorical(y_val_fold)

    model = models[s]
    
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"), 
            tf.keras.metrics.Recall(name="recall"),
            tfa.metrics.F1Score(num_classes,'macro')
]
    if conv2D:
        learning_rate=0.001
    else:
        learning_rate=0.0001

	print("Came here")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics = metrics)
    
    classes = range(num_classes)
    weights = [1,1,1,0.9]

    history = model.fit(X_train_fold,
                        y_train_oh, 
                        batch_size = batch_size, 
                        epochs=epochs, 
                        shuffle = True, 
                        class_weight=None, #dict(zip(classes, weights)),
                        validation_data = (X_val_fold, to_categorical(y_val_fold)),
                        verbose=0)

    model.save(folder_path + "/model_Kfold" + str(s) + ".h5")

  
    # CHECK IF MODEL WAS CORRECTLY SAVED - Commented after checked the saving code is correct
    #reconstructed_model = tf.keras.models.load_model(folder_path + "/model_Kfold" + str(s) + ".h5")

    #np.testing.assert_allclose(
    #    model.predict(X_val_fold), reconstructed_model.predict(X_val_fold)
    #)
    

    val_losses = history.history['val_loss']
    val_accs = history.history['val_accuracy']
    val_pre = history.history['val_precision']
    val_rec = history.history['val_recall']

    for key in train_metrics.keys():
        train_metrics[key].append(history.history[key])
    for key in val_metrics.keys():
        val_metrics[key].append(history.history[key])
    
    score = tf.nn.softmax(model.predict(X_val_fold, verbose=0))
    val_predictions = np.zeros(score.shape[0])
    for t in range(score.shape[0]):
        val_predictions[t] = np.argmax(score[t])
    confusion_matrix = tf.math.confusion_matrix(y_val_fold,val_predictions)
    cf_mat.append(confusion_matrix)

    s = s+1

epochs_range = range(epochs)

# ACCURACY-PRECISION-RECALL
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

for s in range(len(val_metrics["val_accuracy"])):
    acc = np.array(val_metrics["val_accuracy"][s])
    pre = np.array(val_metrics["val_precision"][s])
    rec = np.array(val_metrics["val_recall"][s])
    split_range = epochs_range[:len(val_metrics["val_accuracy"][s])]
    #plt.plot(split_range, val_metrics["val_accuracy"][s], label='Val accuracy, s = ' + str(s+1), alpha = 0.5)
    #plt.plot(split_range, val_metrics["val_recall"][s], label='Val recall, s = ' + str(s+1), alpha = 0.5)
    #plt.plot(split_range, val_metrics["val_precision"][s], label='Val precision, s = ' + str(s+1), alpha = 0.5)
    val_f1 = (2*rec*pre/(pre+rec))
    plt.plot(split_range, val_f1, label='Val f1_score, s = ' + str(s+1))
plt.legend(loc='lower right')
plt.title('Validation Metrics')

plt.subplot(1, 2, 2)
for s in range(len(val_metrics["val_loss"])):
    split_range = epochs_range[:len(val_metrics["val_loss"][s])]
    plt.plot(split_range, val_metrics["val_loss"][s], label= 'Val loss, s = ' + str(s+1))
plt.legend(loc='upper right')
plt.title('Val Loss')
plt.savefig(folder_path + '/val_perf.jpg', dpi = 500)
#plt.show()
plt.close()


#*******************************************************************************************************************************************
# PART-6: PERFORMANCE EVALUATION OVER VALIDATION SET
#****************************************************************************************************************************************
splits_acc = []
splits_pre = []
splits_rec = []
splits_f1 = []

for s, confusion_matrix in enumerate(cf_mat):
    confusion_matrix = confusion_matrix.numpy()
    #print(confusion_matrix.sum(axis=0))
    #print(np.diag(confusion_matrix))

    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP
    TN = confusion_matrix.sum() - (FP + FN + TP)

    #print(FP, FN, TP, TN)

    # Recall
    recall = TP/(TP+FN)
    # Precision
    precision = TP/(TP+FP)
    # Overall accuracy
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    # F1 score
    f1 = 2*precision*recall/(precision+recall)

    
    #print("\n\nSPLIT ", s)

    fig, ax = plt.subplots(figsize = (5,5))
    ax.matshow(confusion_matrix)
    
    colors = ["k", "w"]
    labs = ["ER", "P1", "P4", "P5", "P6","P7"]
    ax.set_xticklabels(['']+labs)
    ax.tick_params(axis = 'x', bottom=False, top = True, labelbottom =False, labeltop = True)
    ax.set_yticklabels(['']+labs)
    ax.tick_params(axis = 'y', right=True, left = False, labelright =True, labelleft = False)
    
    for i in range(num_classes):
        #print(i)
        for j in range(num_classes):
            #print(j)
            #print(confusion_matrix[i, j])
            if confusion_matrix[i, j] > 5:
                color = colors[0]
            else:
                color = colors[1]
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color=color)
    ax.set_xlabel("Predicted classes", fontsize = 18)
    ax.set_ylabel("True classes", fontsize = 18)
    ax.set_title("")
    plt.savefig(folder_path + '/confmat_' + str(s) + '.pdf', dpi = 1000)
    #plt.show()
    plt.close()

    #models[s-1].summary()

    
    print("val recall (per class): ", recall)
    print("val precision (per class): ", precision)
    print("val accuracy (per class): ", accuracy)
    print("val f1-score (per class): ", f1)

    print("Average val recall: ", np.nanmean(recall))
    print("Average val precision: ", np.nanmean(precision))
    print("Average val accuracy: ", np.nanmean(accuracy))
    print("Average val f1: ", np.nanmean(f1))
    

    splits_f1.append(np.nanmean(f1))
    splits_rec.append(np.nanmean(recall))
    splits_pre.append(np.nanmean(precision))
    splits_acc.append(np.nanmean(accuracy))

    val_results = {"acc":accuracy, "pre":precision, "rec":recall,"f1":f1, 
            "avgacc":np.nanmean(accuracy), "avgpre":np.nanmean(precision), "avgrec":np.nanmean(recall),"avgf1":np.nanmean(f1)}
    with open(os.path.join(folder_path,"val_results" + str(s) + ".txt"), "w") as output:
        output.write(str(val_results))

overall_results = {"avgacc":np.nanmean(np.array(splits_acc)), "avgpre":np.nanmean(np.array(splits_pre)), "avgrec":np.nanmean(np.array(splits_rec)),"avgf1":np.nanmean(np.array(splits_f1))}
with open(os.path.join(folder_path,"overall_val_results.txt"), "w") as output:
        output.write(str(overall_results))

print("AVERAGE OVERALL VAL F1 score: ", np.mean(np.array(splits_f1)))
print(np.mean(np.array(splits_f1)))


# TEST RESULTS
final_model = my_model(num_classes=num_classes, input_shape=input_shape)

#********************************************************************************************************************************************************************************
# PART-7:
# Final model is selected from the cross-validation performance
# Prepare train dataset for the selected final model
#*********************************************************************************************************************************************************************************
if mode == "wavelet":
    channels = scales[-1]
elif coeff_aschannels or conv2D:
    channels = 2
else:
    channels = 1

X_train_ = np.zeros([len(X_train), input_len, channels])

for line in range(len(y_train)):   
    temp = np.array(X_train[line])
    
    if mode == "time":
        if coeff_aschannels or conv2D:
            temp = np.reshape(temp, (temp.shape[0],2))
        else:
            temp = np.reshape(temp, (temp.shape[0],1))

    if mode == "fft" or mode == "diff":
        if coeff_aschannels or conv2D:
            temp = np.reshape(temp, (n_fft,2))
        else:
            temp = np.reshape(temp, (temp.shape[0],1))

    if mode == "wavelet":
        sig = np.reshape(temp, (temp.shape[0],))
        coeff, freq = pywt.cwt(sig, scales = scales, wavelet = waveletname, sampling_period = 0.001)
        coeff_ = coeff[:,:len(scales)]   
        temp = coeff_ # np.reshape(coeff_, (coeff_.shape[0], coeff_.shape[1],1))
    
    X_train_[line,:] = temp

    # Normalization before training - in case you add remember to add also for train, val data
    #Xline = X_train_[line,:]
    #X_train_[line,:] = (Xline - np.mean(Xline))/(np.max(Xline) - np.min(Xline))
    #X_train_[line,:] = (Xline - np.mean(Xline))/(np.max(np.abs(Xline)))
    #X_train_[line,:] = (Xline - np.mean(Xline))/(np.max(Xline))
        
    if conv and not coeff_aschannels:
        X_train_ = np.reshape(X_train_, (X_train_.shape[0], X_train_.shape[1], 1))
    
    if conv2D:
        X_train_ = np.reshape(X_train_, (X_train_.shape[0], X_train_.shape[1], X_train_.shape[2], 1))

#**************************************************************************************************************************************************
# PART-8: Prepare test dataset for the selected final model
#***************************************************************************************************************************************************
if mode == "wavelet":
    channels = scales[-1]
elif coeff_aschannels or conv2D:
    channels = 2
else:
    channels = 1

X_test_ = np.zeros([len(X_test), input_len, channels])

for line in range(len(y_test)):   
    temp = np.array(X_test[line])
    
    if mode == "time":
        if coeff_aschannels or conv2D:
            temp = np.reshape(temp, (temp.shape[0],2))
        else:
            temp = np.reshape(temp, (temp.shape[0],1))

    if mode == "fft" or mode == "diff":
        if coeff_aschannels or conv2D:
            temp = np.reshape(temp, (n_fft,2))
        else:
            temp = np.reshape(temp, (temp.shape[0],1))

    if mode == "wavelet":
        sig = np.reshape(temp, (temp.shape[0],))
        coeff, freq = pywt.cwt(sig, scales = scales, wavelet = waveletname, sampling_period = 0.001)
        coeff_ = coeff[:,:len(scales)]   
        temp = coeff_ # np.reshape(coeff_, (coeff_.shape[0], coeff_.shape[1],1))
    
    X_test_[line,:] = temp

    # Normalization before training - in case you add remember to add also for train, val data
    #Xline = X_test_[line,:]
    #X_test_[line,:] = (Xline - np.mean(Xline))/(np.max(Xline) - np.min(Xline))
    #X_test_[line,:] = (Xline - np.mean(Xline))/(np.max(np.abs(Xline)))
    #X_test_[line,:] = (Xline - np.mean(Xline))/(np.max(Xline))
        

if conv and not coeff_aschannels:
    X_test_ = np.reshape(X_test_, (X_test_.shape[0], X_test_.shape[1], 1))
    
if conv2D:
    X_test_ = np.reshape(X_test_, (X_test_.shape[0], X_test_.shape[1], X_test_.shape[2], 1))

print(X_train_.shape)
#******************************************************************************************************************************************************
# PART-9: COMPILE AND FIT the final model
#**************************************************************************************************************************************************************
metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"), 
            tf.keras.metrics.Recall(name="recall"),
            tfa.metrics.F1Score(num_classes,'macro')
]

if conv2D:
    learning_rate=0.001
else:
    learning_rate=0.0001

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics = metrics)

y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

history = final_model.fit(X_train_,
                    y_train_oh, 
                    batch_size = batch_size, 
                    epochs=epochs, 
                    shuffle = True, 
                    class_weight=None, #dict(zip(classes, weights)),
                    verbose=0)

#*************************************************************************************************************************************
# PART-10: Compute test predictions
#*************************************************************************************************************************************
score = tf.nn.softmax(final_model.predict(X_test_, verbose=0))
test_predictions = np.zeros(score.shape[0])
for t in range(score.shape[0]):
    test_predictions[t] = np.argmax(score[t])

# Confusion matrix
confusion_matrix = tf.math.confusion_matrix(y_test,test_predictions)
confusion_matrix = confusion_matrix.numpy()
    
# Compute metrics from TP, FP, TN, FN
TP = np.diag(confusion_matrix)
FP = confusion_matrix.sum(axis=0) - TP
FN = confusion_matrix.sum(axis=1) - TP
TN = confusion_matrix.sum() - (FP + FN + TP)

# Recall
recall = TP/(TP+FN)
# Precision
precision = TP/(TP+FP)
# Overall accuracy
accuracy = (TP+TN)/(TP+FP+FN+TN)
# F1 score
f1 = 2*precision*recall/(precision+recall)

# Show confusion matrix
fig, ax = plt.subplots(figsize = (4,4))
#ax.matshow(confusion_matrix)
ax.matshow(confusion_matrix,cmap=matplotlib.colormaps['YlGn'])    
colors = ["k", "w"]
all_labs = ["ER", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
labs = []

if 8 in considered_positions: 
    labs.append("ER")
for p in considered_positions:
    for l in all_labs:
        if str(p) in l:
            labs.append(l)
        

print(labs)

#ax.set_xticklabels(['']+labs, fontsize = 17)
ax.set_xticklabels(['']+labs, fontsize = 17)
ax.tick_params(axis = 'x', bottom=False, top = True, labelbottom =False, labeltop = True)
#ax.set_yticklabels(['']+labs, fontsize = 17)
ax.set_yticklabels(['']+labs, fontsize = 17)
ax.tick_params(axis = 'y', right=True, left = False, labelright =True, labelleft = False)

for i in range(num_classes):
    for j in range(num_classes):
        if confusion_matrix[i, j] > 3:
            color = colors[1]
        else:
            color = colors[0]
        text = ax.text(j, i, confusion_matrix[i, j],
                    ha="center", va="center", color=color, fontsize=24)
ax.set_xlabel("Predicted classes", fontsize = 24)
ax.set_ylabel("True classes", fontsize = 24)
                   #ha="center", va="center", color=color, fontsize=24)
#ax.set_xlabel("Predicted classes", fontsize = 24)
#ax.set_ylabel("True classes", fontsize = 24)
ax.set_title("")
plt.savefig(folder_path + '/confmat_test.pdf', dpi = 1000)
plt.close()

#****************************************************************************************************************************************
# PART-11: Save test results
#****************************************************************************************************************************************
test_results = {"acc":accuracy, "pre":precision, "rec":recall,"f1":f1, 
        "avgacc":np.nanmean(accuracy), "avgpre":np.nanmean(precision), "avgrec":np.nanmean(recall),"avgf1":np.nanmean(f1)}
with open(os.path.join(folder_path,"test_results.txt"), "w") as output:
    output.write(str(test_results))
