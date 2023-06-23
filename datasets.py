import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re

import os
import matplotlib.pyplot as plt
import sys
import csv

#Replace this path with wherever you saved the PPG Tools folder
sys.path.insert(0, r'ppgtools')

import numpy as np
from scipy import signal
from ppgtools.sigimport import importBIN, importEventMarkers, EventMarker
from ppgtools.biosignal import BioSignal

from ppgtools import sigpro, sigimport, sigplot, biometrics

import scipy.stats as stats
import copy
from sklearn.decomposition import FastICA





class PPGDataset(Dataset):
    '''
        root_dir: full path to data directory
        data_filenames: names of data subdirectories (e.g. ["artery","vein"])
        devicename: "PPG Tattoo v3.2_DC.2B.5A.AE.9E.29" 

        channel_lists: list of desired channels to use from signal (0-3 artery, 4-7 vein, 8-19 accelerometer, 11 temperature)
                       these should be aligned with data_filenames, i.e ["artery","vein"] and [[0,1,2,3],[4,5,6,7]]
        window_len: length of a window in samples (sampling rate is 25 Hz)
        overlap_frac: how much consecutive windows overlap, e.g. 0.5 means 50% overlap
        rescale: an interval to rescale the data, e.g. [-1,1]
        normalize: make zero mean and unit variance
    '''

    # filenames should be a list so we can load multiple and normalize/rescale together
    def __init__(self, root_dir,data_filenames,devicename,channel_lists,window_len,overlap_frac,rescale=None,normalize=False,fc=False):
        # get the label (for now assume artery = 1, vein = 0)
        # if channels == [0,1,2,3]:
        #     self.label = 1
        # elif channels == [4,5,6,7]:
        #     self.label = 0

        # save member variables
        self.data_filenames = data_filenames
        self.channel_lists = channel_lists
        self.window_len = window_len
        self.overlap_frac = overlap_frac
        self.rescale = rescale
        self.normalize = normalize
        self.fc = fc

        # load the data
        data_arrays = []
        label_lists = []
        self.source_list = []
        for df_name,ch_list in zip(data_filenames,channel_lists):
            if ch_list == [0,1,2,3]: # artery
                label = 1
            elif ch_list == [4,5,6,7]: # vein
                label = 0
            sessionData = sigimport.importTattooData(root_dir, df_name)
            signals_original = sessionData[devicename]["Data"]
            markers = sessionData[devicename]["Markers"]  

            # save names for desired signal channels
            signals = [signals_original[channel] for channel in ch_list]
            self.channel_name_map = {i:signal.name for i,signal in enumerate(signals)}

            # copy into a (C x N) numpy array
            num_channels = len(ch_list)
            num_samples = len(signals_original[0].data)
            data_array = np.zeros((num_channels,num_samples))
            label_list = np.zeros(num_samples)
            
            label_list[:] = label
            for sample in range(num_samples):
                self.source_list.append(df_name)

            for ch_i,channel in enumerate(ch_list):
                data_array[ch_i,:] = signals_original[channel].data[:]
            
            data_arrays.append(data_array)
            label_lists.append(label_list)


        # merge the two
        self.data_array = np.concatenate(data_arrays,axis=1)
        self.label_list = np.concatenate(label_lists).astype(np.int64)

        # preprocess the channels if applicable
        if self.rescale is not None:
            for channel_i,signal_data in enumerate(self.data_array):
                self.data_array[channel_i,:] = ((signal_data-min(signal_data))/(max(signal_data)-min(signal_data)))*(rescale[1]-rescale[0]) + rescale[0]

        if self.normalize == True:
            for channel_i,signal_data in enumerate(self.data_array):
                self.data_array[channel_i,:] = (signal_data-signal_data.mean())/(signal_data.std() + 1e-6)

    def __getitem__(self, index):
        # the index represents a window index, we need to return the data and label

        # get the starting and ending sample indices
        start_idx = int((1-self.overlap_frac)*self.window_len)*index
        end_idx = start_idx + self.window_len

        # extract the window (get all channels)
        X = self.data_array[:,start_idx:end_idx]
        if self.fc == True:
            X = X.flatten()
        Y = self.label_list[start_idx]

        return torch.tensor(X,dtype=torch.float),Y
    
    def __len__(self):
        num_windows = (self.data_array.shape[1] - self.window_len)//int((1-self.overlap_frac)*self.window_len) + 1
        return num_windows
    
    def visualize_samples(self):
        matplotlib.rcParams.update({'font.size': 10})
        idxs = torch.randperm(len(self))[:6]
        fig,ax = plt.subplots(2,3,figsize=(12,6),sharey=True)
        fig.subplots_adjust(wspace=0.4,hspace=0.5)
        for i,idx in enumerate(idxs):
            ppg_data,l = self.__getitem__(idx)
            ir = ppg_data[0,:self.window_len]
            red = ppg_data[1,:self.window_len]
            green = ppg_data[2,:self.window_len]
            amb = ppg_data[3,:self.window_len]

            i_x = i % 3
            i_y = i // 3
            x_ = np.arange(self.window_len)
            ax[i_y,i_x].plot(x_,ir,label='ir',c='k')
            ax[i_y,i_x].plot(x_,green,label='green',c='g')
            ax[i_y,i_x].plot(x_,red,label='red',c='r')
            ax[i_y,i_x].plot(x_,amb,label='ambient',c='b')
            ax[i_y,i_x].set_xlabel("Sample #")
            ax[i_y,i_x].set_ylabel("Value")
            source = self.source_list[idx*self.window_len]
            if source == r"\artery":
                id = 0
            elif source == r"\vein":
                id = 1
            elif source == r"\double":
                if l == 1:
                    id = 0
                if l == 0:
                    id = 1
            ax[i_y,i_x].set_title(f"{source[1:]}, ch: [{self.channel_lists[id][0]}-{self.channel_lists[id][-1]}], y: {l}")
        ax[i_y,i_x].legend(loc=(1.05,0.1))

        plt.show()




def load_ppg_dataset(batch_size,window_len,overlap_frac,train_frac,val_frac,split_seed,rescale=None,normalize=False,fc=False):
    # path = r"Data" #Put the directory you have the data in
    # filenames = [r"\artery"] #Outer folder name of dataset (e.g., artery, vein, double)
    # devicename = "PPG Tattoo v3.2_DC.2B.5A.AE.9E.29"   #Don't change this

    # datasets = []
    # datasets.append(PPGDataset(path,r"\artery",devicename,[0,1,2,3],window_len,overlap_frac,rescale=rescale,normalize=normalize))
    # datasets.append(PPGDataset(path,r"\vein",devicename,[4,5,6,7],window_len,overlap_frac,rescale=rescale,normalize=normalize))
    # # for fn in filenames:
    # #     datasets.append(PPGDataset(path,fn,devicename,[0,1,2,3],window_len,overlap_frac,rescale=rescale,normalize=normalize))
    # #     datasets.append(PPGDataset(path,fn,devicename,[4,5,6,7],window_len,overlap_frac,rescale=rescale,normalize=normalize))

    # dataset = torch.utils.data.ConcatDataset(datasets)

    if train_frac > 0:
        filenames = [r"\artery",r"\vein"] #train on vein and artery
    else:
        filenames = [r"\double",r"\double"] #test on double

    path = r"Data" #Put the directory you have the data in
    devicename = "PPG Tattoo v3.2_DC.2B.5A.AE.9E.29"   #Don't change this

    dataset = PPGDataset(path,filenames,devicename,[[0,1,2,3],[4,5,6,7]],50,0,rescale,normalize,fc=fc)

    if train_frac == 0:
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    
    num_train_total = int(train_frac*len(dataset)) # train-test split
    num_test = len(dataset) - num_train_total
    num_val = int(val_frac*num_train_total) # train-val split
    num_train = num_train_total - num_val

    # # since this is time series data, let's split into disjoint blocks (need to ensure equal class distribution)
    # # select a contiguous block for testing, then one for validation, and use the remaining samples for training
    # # add padding between blocks to prevent overlap
    # # -----------------------------------------------------
    # # |   train   || test || train || validation || train |
    # # -----------------------------------------------------

    # padding = 2 # leave two windows unused between each block

    # # get random test block
    # test_start = torch.randperm(len(dataset))[0]
    # test_end = test_start + num_test
    # if test_end > len(dataset): # if extends beyond end, shift accordingly
    #     test_start = len(dataset) - num_test
    #     test_end = len(dataset)

    # # get random validation block
    # val_start = torch.randperm(len(dataset))[0]
    # val_end = val_start + num_val
    # if val_end > len(dataset): # if extends beyond end, shift accordingly
    #     val_start = len(dataset) - num_val
    #     val_end = len(dataset)
    # if val_end > test_start and val_end < test_end: # if right side overlaps with test, shift accordingly
    #     val_start = test_start - num_val - padding
    #     val_end = test_start - padding
    #     if val_start < 0: # if shift beyond start, move to other side of test block
    #         val_start = test_end + padding
    #         val_end = val_start + num_val
    # elif val_start > test_start and val_start < test_end: # if left side overlaps with test, shift accordingly
    #     val_start = test_end + padding
    #     val_end = val_start + num_val
    #     if val_end > len(dataset): # if shift beyond end, move to other side of test block
    #         val_start = test_start - num_val - padding
    #         val_end = test_start - padding
    
    # # use remaining indices for training
    # test_indices = torch.arange(test_start,test_end)
    # val_indices = torch.arange(val_start,val_end)
    # test_val_indices

    print(f"num_train:{num_train}, num_test:{num_test},num_val:{num_val}")
    dataset_lens = [(dataset.label_list == 1).sum()//window_len,(dataset.label_list == 0).sum()//window_len]
    print(f"dataset_lens:{dataset_lens}")
    all_indices = torch.arange(sum(dataset_lens))

    ds1_test_indices = torch.arange(int(dataset_lens[0]*train_frac),dataset_lens[0])
    ds2_test_indices = dataset_lens[0] + torch.arange(int(dataset_lens[1]*train_frac),dataset_lens[1])
    # ds1_test_indices = torch.arange(0,int(dataset_lens[0]*(1-train_frac)))
    # ds2_test_indices = dataset_lens[0] + torch.arange(0,int(dataset_lens[1]*(1-train_frac)))

    ds1_val_indices = torch.arange(int(dataset_lens[0]*train_frac*val_frac),int(dataset_lens[0]*train_frac))
    ds2_val_indices = dataset_lens[0] + torch.arange(int(dataset_lens[1]*train_frac*val_frac),int(dataset_lens[1]*train_frac))
    # ds1_val_indices = torch.arange(int(dataset_lens[0]*(1-train_frac)), int(dataset_lens[0]*(1-train_frac)) + int(dataset_lens[0]*(train_frac*val_frac)))
    # ds2_val_indices = dataset_lens[0] + torch.arange(int(dataset_lens[1]*(1-train_frac)), int(dataset_lens[1]*(1-train_frac)) + int(dataset_lens[1]*(train_frac*val_frac)))


    test_indices = all_indices[torch.cat([ds1_test_indices,ds2_test_indices])]
    val_indices = all_indices[torch.cat([ds1_val_indices,ds2_val_indices])]
    train_indices = torch.from_numpy(np.setdiff1d(all_indices.numpy(),np.concatenate([val_indices.numpy(),test_indices.numpy()])))

    print(f"train_len:{len(train_indices)}")
    print(f"val_len:{len(val_indices)}")
    print(f"test_len:{len(test_indices)}")

    print(f"train_indices: {train_indices}")
    print(f"val_indices: {val_indices}")
    print(f"test_indices: {test_indices}")


    # train-test then train-val splits using a seed
    test_split = torch.utils.data.Subset(dataset,test_indices)
    val_split = torch.utils.data.Subset(dataset,val_indices)
    train_split = torch.utils.data.Subset(dataset,train_indices)
    # return train_split,test_split
    # train_split, test_split = torch.utils.data.random_split(dataset, [num_train_total, num_test],torch.Generator().manual_seed(split_seed))
    # train_split, val_split = torch.utils.data.random_split(train_split, [num_train, num_val],torch.Generator().manual_seed(split_seed))
    
    # create the dataloaders, use a seed for sample ordering
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_split, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader,test_loader



def ppg_parser(batch,device):
    # get the data and target
    data, target = batch[0].to(device), batch[1].to(device)

    return data, target

class PPG_LOSS(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, out, target):
    # return cross entropy
    return torch.nn.CrossEntropyLoss()(out,target)
  
# similar for evaluation
class PPG_EVAL():
  def __call__(self, out, target):
    # get batch predictions
    prediction = out.argmax(dim=1).to('cpu')
    
    return prediction,target



if __name__ == '__main__':
    # train_loader, val_loader, test_loader = load_ppg_dataset(16,50,0,0.8,0.1,1234,None,True)
    path = r"Data" #Put the directory you have the data in
    # filenames = [r"\artery",r"\vein"] #Outer folder name of dataset (e.g., artery, vein, double)
    filenames = [r"\double",r"\double"]
    devicename = "PPG Tattoo v3.2_DC.2B.5A.AE.9E.29"   #Don't change this
    dataset = PPGDataset(path,filenames,devicename,[[0,1,2,3],[4,5,6,7]],50,0)
    # print(dataset[0])
    dataset.visualize_samples()
    # print((dataset.label_list == 0).sum())
    # print(len(dataset))
    # print(dataset.label_list)