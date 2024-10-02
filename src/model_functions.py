from Layers import GravNet_simple, GlobalExchange, GravNet
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import math
import time
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,Embedding
from tensorflow.keras.layers import BatchNormalization,Concatenate, Lambda
from tensorflow.keras.layers import concatenate, Lambda
import tensorflow.keras as keras
import tensorflow as tf
import pickle
from scipy.optimize import curve_fit
K = keras.backend

def make_gravnet_model(K=20, N_feat=23, N_grav_layers=2, N_neighbors=10, N_filters=256, use_sector=True):
    inputs = Input(shape=(K, N_feat), name='input1')
    
    # Trim off the last 6 features if use_sector is False
    if not use_sector:
        x = inputs[:, :, :-6]  # Trim the last 6 features
    else:
        x = inputs  # Use the original full feature set
    
    if type(N_neighbors)!=list:
        N_neighbors = [N_neighbors] * N_grav_layers
    else:
        assert(len(N_neighbors)==N_grav_layers)
    #v = Dense(64, activation='elu',name='Dense0')(m_input)

    #x = BatchNormalization(momentum=0.6,name='batchNorm1')(x)
    
    feat=[x]

    for i in range(N_grav_layers):#12 or 6
        #x = GlobalExchange(name='GE_l'+str(i))(x)
        #v = Dense(64, activation='elu',name='Dense0_l'+str(i))(v)
        #x = BatchNormalization(momentum=0.6,name='batchNorm1_l'+str(i))(x)
        x = Dense(64, activation='elu',name='Dense1_l'+str(i))(x)
        x = GravNet(n_neighbours=N_neighbors[i],#10 
                       n_dimensions=4, #4
                       n_filters=N_filters,#128 or 256
                       n_propagate=32,
                       subname='GNet_l'+str(i),
                       name='GNet_l'+str(i))(x)#or inputs??#64 or 128
        #x = BatchNormalization(momentum=0.6,name='batchNorm2_l'+str(i))(x)
        # x = Dropout(0.1,name='dropout_l'+str(i))(x) #test
        feat.append(Dense(32, activation='elu',name='Dense2_l'+str(i))(x))

    x = Concatenate(name='concat1')(feat)
    
    x = Dense(32, activation='elu', name='Dense3')(x)
    out_beta = Dense(1, activation='sigmoid', name='out_beta')(x)
    out_latent = Dense(2, name='out_latent')(x)
    out=concatenate([out_beta, out_latent])

    model=keras.Model(inputs=inputs, outputs=out)
    
    return model