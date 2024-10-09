from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Activation
import tensorflow.keras as keras
import tensorflow as tf
from Layers import GravNet

# Custom function to clip the values between 0.0001 and 0.99999
def clip_out_beta(y):
    return tf.clip_by_value(y, clip_value_min=0.0001, clip_value_max=0.99999)


def make_gravnet_model(K=20, N_feat=23, N_grav_layers=2, N_neighbors=10, N_filters=256, use_sector=True):
    inputs = Input(shape=(K, N_feat), name='input1')
    
    # Trim off the last 6 features if use_sector is False
    if not use_sector:
        x = inputs[:, :, :-6]  # Trim the last 6 features
    else:
        x = inputs  # Use the original full feature set
    
    if type(N_neighbors) != list:
        N_neighbors = [N_neighbors] * N_grav_layers
    else:
        assert len(N_neighbors) == N_grav_layers

    feat = [x]

    for i in range(N_grav_layers):
        x = Dense(64, activation='elu', name='Dense1_l' + str(i))(x)
        x = GravNet(n_neighbours=N_neighbors[i],
                    n_dimensions=4,
                    n_filters=N_filters,
                    n_propagate=32,
                    subname='GNet_l' + str(i),
                    name='GNet_l' + str(i))(x)
        feat.append(Dense(32, activation='elu', name='Dense2_l' + str(i))(x))

    x = Concatenate(name='concat1')(feat)
    
    x = Dense(128, activation='elu', name='Dense3')(x)
    x = Dense(64, activation='elu', name='Dense4')(x)
    
    # Use the named function instead of the Lambda layer for clipping
    out_beta = Dense(1, activation='sigmoid', name='out_beta')(x)
    out_beta = tf.keras.layers.Activation(clip_out_beta, name='clip_out_beta')(out_beta)
    
    out_latent = Dense(2, name='out_latent')(x)
    out_pid = Dense(3, activation='softmax', name='out_pid')(x)
    
    out = Concatenate()([out_beta, out_latent, out_pid])

    model = keras.Model(inputs=inputs, outputs=out)
    
    return model