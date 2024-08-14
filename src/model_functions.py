from Layers import GravNet_simple, GlobalExchange
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import math
import time
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,Embedding
from tensorflow.keras.layers import BatchNormalization,Concatenate, Lambda
from tensorflow.keras.layers import concatenate
import tensorflow.keras as keras
import tensorflow as tf
import pickle
from scipy.optimize import curve_fit
K = keras.backend

def make_gravnet_model(K=20):
    m_input = Input(shape=(K,23,),name='input1')#dtype=tf.float64

    #v = Dense(64, activation='elu',name='Dense0')(m_input)

    #v = BatchNormalization(momentum=0.6,name='batchNorm1')(inputs)
    
    feat=[m_input]
    
    for i in range(2):#12 or 6
        v = GlobalExchange(name='GE_l'+str(i))(m_input)
        #v = Dense(64, activation='elu',name='Dense0_l'+str(i))(v)
        #v = BatchNormalization(momentum=0.6,name='batchNorm1_l'+str(i))(v)
        v = Dense(64, activation='elu',name='Dense1_l'+str(i))(v)
        v = GravNet_simple(n_neighbours=4,#10 
                       n_dimensions=4, #4
                       n_filters=256,#128 or 256
                       n_propagate=32,
                       name='GNet_l'+str(i))(v)#or inputs??#64 or 128
        v = BatchNormalization(momentum=0.6,name='batchNorm2_l'+str(i))(v)
        v = Dropout(0.2,name='dropout_l'+str(i))(v) #test
        feat.append(Dense(32, activation='elu',name='Dense2_l'+str(i))(v))

    v = Concatenate(name='concat1')(feat)
    
    v = Dense(32, activation='elu',name='Dense3')(v)
    out_beta=Dense(1,activation='sigmoid',name='out_beta')(v)
    out_latent=Dense(2,name='out_latent')(v)
    out=concatenate([out_beta, out_latent])

    model=keras.Model(inputs=m_input, outputs=out)
    
    return model