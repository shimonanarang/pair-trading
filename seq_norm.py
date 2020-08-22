import pandas as pd 
import numpy as np

def seq_and_norm(data, seq_len):
    print(data.shape)
    start = 0
    sequence_x = []
    target = []
    for i in range(len(data)-seq_len):
        x = data.iloc[start:start+seq_len,:]
        y = data.iloc[start+1:start+1+seq_len,0] # assuming 1st element in the row is target
        
        #window normalization
        x = x/x.iloc[0,:] -1 
        y = y/y.iloc[0] -1

        #append in the sequence and target series
        sequence_x.append(x.values)
        target.append(y.values)
        
        start += 1
    #return array with size [batch+size, seq_len, num_features]
    #target is [batch_size, seq_len]
    return np.array(sequence_x), np.array(target)
    

    