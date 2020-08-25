import pandas as pd 
import numpy as np

def seq_and_norm(data, seq_len):
    print(data.shape)
    start = 0
    sequence_x = []
    target = []
    initial_y =  []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    for i in range(len(data)-seq_len):
        x = data.iloc[start:start+seq_len,:]                                                                                                                                                                                                                                                                      
        y = data.iloc[start+1:start+1+seq_len,0] # assuming 1st element in the row is target
        
        #window normalization
        y_0 = y.iloc[0]            
        x = x/x.iloc[0,:] -1 
        y = y/y.iloc[0] -1
        
        initial_y.append([y_0]*seq_len) #this will be used for inverse_transform function

        #append in the sequence and target series
        sequence_x.append(x.values)
        target.append(y.values)                                                                                                                                                                                                                                                                                                                                                                                           
        
        start += 1
    #return array with size [batch+size, seq_len, num_features]
    #target is [batch_size, seq_len]
    return np.array(sequence_x), np.array(target), np.array(initial_y)


def inverse_transform(y, initial_y):
    #if 3-D 
    if len(y.shape) == 3:
        y = np.multiply(y[:,:,0],initial_y[:,0]) + initial_y
    #if 2-D (for predicted y)
    if len(y.shape) == 2:
        y = np.multiply(y[:,0],initial_y[:,0]) + initial_y[:,0]
    return(y)
