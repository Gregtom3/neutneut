
global_layers_list = {}

from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

from caloGraphNN import *
from caloGraphNN_keras import GlobalExchange
global_layers_list['GlobalExchange']=GlobalExchange  

# Define custom layers here and add them to the global_layers_list dict (important!)


class Conv2DGlobalExchange(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(Conv2DGlobalExchange, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+input_shape[3])
    
    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        average = tf.tile(average, [1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])
        return tf.concat([inputs,average],axis=-1)
        
    
    def get_config(self):
        base_config = super(Conv2DGlobalExchange, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['Conv2DGlobalExchange']=Conv2DGlobalExchange 


class PadTracker(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(PadTracker, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],2*32,2*32,input_shape[3])
    
    def call(self, inputs):
        return tf.pad(inputs, [[0,0],[16,16],[16,16],[0,0]])
        
    
    def get_config(self):
        base_config = super(PadTracker, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['PadTracker']=PadTracker 


class CropTracker(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(CropTracker, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],32,32,input_shape[3])
    
    def call(self, inputs):
        return inputs[:,16:48,16:48,:]
        
    
    def get_config(self):
        base_config = super(CropTracker, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['CropTracker']=CropTracker 


class TileTrackerFeatures(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(TileTrackerFeatures, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]*4)
    
    def call(self, inputs):
        return tf.tile(inputs, [1,1,1,4])
        
    
    def get_config(self):
        base_config = super(TileTrackerFeatures, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['TileTrackerFeatures']=TileTrackerFeatures 



class TileCalo(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(TileCalo, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4*16,4*16,input_shape[3])
    
    def call(self, inputs):
        return tf.tile(inputs, [1,4,4,1])
        
    
    def get_config(self):
        base_config = super(TileCalo, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['TileCalo']=TileCalo 

class Tile2D(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, ntiles, **kwargs):
        super(Tile2D, self).__init__(**kwargs)
        self.ntiles=ntiles
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4*16,4*16,input_shape[3])
    
    def call(self, inputs):
        return tf.tile(inputs, [1,self.ntiles,self.ntiles,1])
        
    
    def get_config(self):
        base_config = super(Tile2D, self).get_config()
        config = {'ntiles' :self.ntiles}
        return dict(list(config.items() ) + list(base_config.items()) )
        

global_layers_list['Tile2D']=Tile2D 


class GaussActivation(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(GaussActivation, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.exp(- inputs**2 )
        
    
    def get_config(self):
        base_config = super(GaussActivation, self).get_config()
        return dict(list(base_config.items()) )
        

global_layers_list['GaussActivation']=GaussActivation 






# class GravNet_simple(tf.keras.layers.Layer):
#     def __init__(self, 
#                  n_neighbours, 
#                  n_dimensions, 
#                  n_filters, 
#                  n_propagate,**kwargs):
#         super(GravNet_simple, self).__init__(**kwargs)
        
#         self.n_neighbours = n_neighbours
#         self.n_dimensions = n_dimensions
#         self.n_filters = n_filters
#         self.n_propagate = n_propagate
        
#         self.input_feature_transform = tf.keras.layers.Dense(n_propagate)
#         self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions)
#         self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh')

#     def build(self, input_shape):
        
#         self.input_feature_transform.build(input_shape)
#         self.input_spatial_transform.build(input_shape)
        
#         self.output_feature_transform.build((input_shape[0], input_shape[1], 
#                                              input_shape[2] + self.input_feature_transform.units * 2))
 
#         super(GravNet_simple, self).build(input_shape)
        
#     def call(self, x):
        
#         coordinates = self.input_spatial_transform(x)
#         features = self.input_feature_transform(x)
#         collected_neighbours = self.collect_neighbours(coordinates, features)
        
#         updated_features = tf.concat([x, collected_neighbours], axis=-1)
#         return self.output_feature_transform(updated_features)
    

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1], self.output_feature_transform.units)
    
#     def collect_neighbours(self, coordinates, features):
        
#         distance_matrix = euclidean_squared(coordinates, coordinates)
#         ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

#         neighbour_indices = ranked_indices[:, :, 1:]
        
#         n_batches = tf.shape(features)[0]
#         n_vertices = tf.shape(features)[1]
#         n_features = tf.shape(features)[2]
        
#         batch_range = tf.range(0, n_batches)
#         batch_range = tf.expand_dims(batch_range, axis=1)
#         batch_range = tf.expand_dims(batch_range, axis=1)
#         batch_range = tf.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

#         # tf.ragged FIXME? n_vertices
#         batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
#         vertex_indices = tf.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
#         indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
    
#         distance = -ranked_distances[:, :, 1:]
    
#         weights = gauss_of_lin(distance * 10.)
#         weights = tf.expand_dims(weights, axis=-1)
        
#         neighbour_features = tf.gather_nd(features, indices)
#         neighbour_features *= weights
#         neighbours_max = tf.reduce_max(neighbour_features, axis=2)
#         neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
        
#         return tf.concat([neighbours_max, neighbours_mean], axis=-1)
    
#     def get_config(self):
#         config = {'n_neighbours': self.n_neighbours, 
#                   'n_dimensions': self.n_dimensions, 
#                   'n_filters': self.n_filters, 
#                   'n_propagate': self.n_propagate}
#         base_config = super(GravNet_simple, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))



# class GravNet(tf.keras.layers.Layer):
#     def __init__(self, 
#                  n_neighbours, 
#                  n_dimensions, 
#                  n_filters, 
#                  n_propagate, 
#                  subname, 
#                  also_coordinates=False, feature_dropout=-1, 
#                  coordinate_kernel_initializer=tf.keras.initializers.Orthogonal(),
#                  other_kernel_initializer='glorot_uniform',
#                  fix_coordinate_space=False, 
#                  coordinate_activation=None,
#                  masked_coordinate_offset=None,
#                  additional_message_passing=0,
#                  **kwargs):
#         super(GravNet, self).__init__(**kwargs)

#         self.n_neighbours = n_neighbours
#         self.n_dimensions = n_dimensions
#         self.n_filters = n_filters
#         self.n_propagate = n_propagate
        
#         self.subname = subname
#         name = self.name+subname
        
#         self.also_coordinates = also_coordinates
#         self.feature_dropout = feature_dropout
#         self.masked_coordinate_offset = masked_coordinate_offset
#         self.additional_message_passing = additional_message_passing
        
#         self.input_feature_transform = tf.keras.layers.Dense(n_propagate, name = name+'_FLR', kernel_initializer=other_kernel_initializer)
#         self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions, name = name+'_S', kernel_initializer=coordinate_kernel_initializer, activation=coordinate_activation)
#         self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh', name = name+'_Fout', kernel_initializer=other_kernel_initializer)

#         self._sublayers = [self.input_feature_transform, self.input_spatial_transform, self.output_feature_transform]
#         if fix_coordinate_space:
#             self.input_spatial_transform = None
#             self._sublayers = [self.input_feature_transform, self.output_feature_transform]
        
#         self.message_passing_layers = []
#         for i in range(additional_message_passing):
#             self.message_passing_layers.append(
#                 tf.keras.layers.Dense(n_propagate, activation='elu', 
#                                    name = name+'_mp_'+str(i), kernel_initializer=other_kernel_initializer)
#                 )
#         if additional_message_passing>0:
#             self._sublayers += self.message_passing_layers


#     def build(self, input_shape):
#         if self.masked_coordinate_offset is not None:
#             input_shape = input_shape[0]
            
#         self.input_feature_transform.build(input_shape)
#         if self.input_spatial_transform is not None:
#             self.input_spatial_transform.build(input_shape)
        
#         # tf.ragged FIXME?
#         self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.input_feature_transform.units * 2))

#         self.message_parsing_distance_weights=[]
#         for i in range(len(self.message_passing_layers)):
#             l = self.message_passing_layers[i]
#             l.build((input_shape[0], input_shape[1], input_shape[2] + self.n_propagate * 2))
#             d_weight = self.add_weight(name=self.subname+'_mp_distance_weight_'+str(i), 
#                                       shape=(1,),
#                                       initializer='uniform',
#                                       constraint=tf.keras.constraints.NonNeg() ,
#                                       trainable=True)
#             self.message_parsing_distance_weights.append(d_weight)

#         for layer in self._sublayers:
#             self._trainable_weights.extend(layer.trainable_weights)
#             self._non_trainable_weights.extend(layer.non_trainable_weights)
        
#         super(GravNet, self).build(input_shape)

#     def call(self, x):
        
#         mask = None
#         if self.masked_coordinate_offset is not None:
#             if not isinstance(x, list):
#                 raise Exception('GravNet: in mask mode, input must be list of input,mask')
#             mask = x[1]
#             x = x[0]
            
        
#         if self.input_spatial_transform is not None:
#             coordinates = self.input_spatial_transform(x)
#         else:
#             coordinates = x[:,:,0:self.n_dimensions]
            
#         if self.masked_coordinate_offset is not None:
#             sel_mask = tf.tile(mask, [1,1,tf.shape(coordinates)[2]])
#             coordinates = tf.where(sel_mask>0., coordinates, tf.zeros_like(coordinates)-self.masked_coordinate_offset)

#         collected_neighbours = self.collect_neighbours(coordinates, x, mask)

#         updated_features = tf.concat([x, collected_neighbours], axis=-1)
#         output = self.output_feature_transform(updated_features)
        
#         if self.masked_coordinate_offset is not None:
#             output *= mask

#         if self.also_coordinates:
#             return [output, coordinates]
#         return output
        

#     def compute_output_shape(self, input_shape):
#         if self.masked_coordinate_offset is not None:
#             input_shape = input_shape[0]
#         if self.also_coordinates:
#             return [(input_shape[0], input_shape[1], self.output_feature_transform.units),
#                     (input_shape[0], input_shape[1], self.n_dimensions)]
        
#         # tf.ragged FIXME? tf.shape() might do the trick already
#         return (input_shape[0], input_shape[1], self.output_feature_transform.units)

#     def collect_neighbours(self, coordinates, x, mask):
        
#         # tf.ragged FIXME?
#         # for euclidean_squared see caloGraphNN.py
#         distance_matrix = euclidean_squared(coordinates, coordinates)
#         ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

#         neighbour_indices = ranked_indices[:, :, 1:]
        
#         features = self.input_feature_transform(x)
        
#         n_batches = tf.shape(features)[0]
        
#         # tf.ragged FIXME? or could that work?
#         n_vertices = tf.shape(features)[1]
#         n_features = tf.shape(features)[2]

#         batch_range = tf.range(0, n_batches)
#         batch_range = tf.expand_dims(batch_range, axis=1)
#         batch_range = tf.expand_dims(batch_range, axis=1)
#         batch_range = tf.expand_dims(batch_range, axis=1) # (B, 1, 1, 1)

#         # tf.ragged FIXME? n_vertices
#         batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1]) # (B, V, N-1, 1)
#         vertex_indices = tf.expand_dims(neighbour_indices, axis=3) # (B, V, N-1, 1)
#         indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
    
#         distance = -ranked_distances[:, :, 1:]
    
#         weights = gauss_of_lin(distance * 10.)
#         weights = tf.expand_dims(weights, axis=-1)
        
    
#         for i in range(len(self.message_passing_layers)+1):
#             if i:
#                 features = self.message_passing_layers[i-1](tf.concat([features,x],axis=-1))
#                 w=self.message_parsing_distance_weights[i-1]
#                 weights = gauss_of_lin(w*distance)
#                 weights = tf.expand_dims(weights, axis=-1)
                
#             if self.feature_dropout>0 and self.feature_dropout < 1:
#                 features = tf.keras.layers.Dropout(self.feature_dropout)(features)
                
#             neighbour_features = tf.gather_nd(features, indices) # (B, V, N-1, F)
#             # weight the neighbour_features
#             neighbour_features *= weights
            
#             neighbours_max = tf.reduce_max(neighbour_features, axis=2)
#             neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
            
#             features = tf.concat([neighbours_max, neighbours_mean], axis=-1)
#             if mask is not None:
#                 features *= mask
        
#         return features

#     def get_config(self):
#             config = {'n_neighbours': self.n_neighbours, 
#                       'n_dimensions': self.n_dimensions, 
#                       'n_filters': self.n_filters, 
#                       'n_propagate': self.n_propagate,
#                       'subname':self.subname,
#                       'also_coordinates': self.also_coordinates,
#                       'feature_dropout' : self.feature_dropout,
#                       'masked_coordinate_offset'       : self.masked_coordinate_offset,
#                       'additional_message_passing'    : self.additional_message_passing}
#             base_config = super(GravNet, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
        
# global_layers_list['GravNet_simple']=GravNet_simple   
# global_layers_list['GravNet']=GravNet 




class GravNet_simple(tf.keras.layers.Layer):
    def __init__(self, 
                 n_neighbours, 
                 n_dimensions, 
                 n_filters, 
                 n_propagate, **kwargs):
        super(GravNet_simple, self).__init__(**kwargs)
        
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        
        self.input_feature_transform = tf.keras.layers.Dense(n_propagate)
        self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions)
        self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh')

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.input_spatial_transform.build(input_shape)
        self.output_feature_transform.build((input_shape[0], input_shape[1], 
                                             input_shape[2] + self.input_feature_transform.units * 2))
        super(GravNet_simple, self).build(input_shape)
        
    def call(self, x):
        coordinates = self.input_spatial_transform(x)
        features = self.input_feature_transform(x)
        collected_neighbours = self.collect_neighbours(coordinates, features)
        
        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        return self.output_feature_transform(updated_features)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)
    
    def collect_neighbours(self, coordinates, features):
        distance_matrix = euclidean_squared(coordinates, coordinates)
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]
        n_batches = tf.shape(features)[0]
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]

        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)  # (B, 1, 1, 1)

        batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1])  # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices, axis=3)  # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
        distance = -ranked_distances[:, :, 1:]
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)
        
        neighbour_features = tf.gather_nd(features, indices)
        neighbour_features *= weights
        neighbours_max = tf.reduce_max(neighbour_features, axis=2)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)
        
        return tf.concat([neighbours_max, neighbours_mean], axis=-1)
    
    def get_config(self):
        config = {'n_neighbours': self.n_neighbours, 
                  'n_dimensions': self.n_dimensions, 
                  'n_filters': self.n_filters, 
                  'n_propagate': self.n_propagate}
        base_config = super(GravNet_simple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GravNet(tf.keras.layers.Layer):
    def __init__(self, 
                 n_neighbours, 
                 n_dimensions, 
                 n_filters, 
                 n_propagate, 
                 subname, 
                 also_coordinates=False, feature_dropout=-1, 
                 coordinate_kernel_initializer=tf.keras.initializers.Orthogonal(),
                 other_kernel_initializer='glorot_uniform',
                 fix_coordinate_space=False, 
                 coordinate_activation=None,
                 masked_coordinate_offset=None,
                 additional_message_passing=0,
                 **kwargs):
        super(GravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.subname = subname
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.masked_coordinate_offset = masked_coordinate_offset
        self.additional_message_passing = additional_message_passing
        
        name = self.name + subname

        self.input_feature_transform = tf.keras.layers.Dense(n_propagate, name=name+'_FLR', kernel_initializer=other_kernel_initializer)
        self.input_spatial_transform = tf.keras.layers.Dense(n_dimensions, name=name+'_S', kernel_initializer=coordinate_kernel_initializer, activation=coordinate_activation)
        self.output_feature_transform = tf.keras.layers.Dense(n_filters, activation='tanh', name=name+'_Fout', kernel_initializer=other_kernel_initializer)

        self.message_passing_layers = [
            tf.keras.layers.Dense(n_propagate, activation='elu', name=name+'_mp_'+str(i), kernel_initializer=other_kernel_initializer)
            for i in range(additional_message_passing)
        ]
        self.message_parsing_distance_weights = [
            self.add_weight(name=subname+'_mp_distance_weight_'+str(i), shape=(1,), initializer='uniform', constraint=tf.keras.constraints.NonNeg(), trainable=True)
            for i in range(additional_message_passing)
        ]

    def build(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        
        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)
        
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.input_feature_transform.units * 2))

        for i, layer in enumerate(self.message_passing_layers):
            layer.build((input_shape[0], input_shape[1], input_shape[2] + self.n_propagate * 2))
        
        super(GravNet, self).build(input_shape)

    def call(self, x):
        mask = None
        if self.masked_coordinate_offset is not None:
            if not isinstance(x, list):
                raise Exception('GravNet: in mask mode, input must be list of input,mask')
            mask = x[1]
            x = x[0]
        
        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:, :, :self.n_dimensions]

        if self.masked_coordinate_offset is not None:
            sel_mask = tf.tile(mask, [1, 1, tf.shape(coordinates)[2]])
            coordinates = tf.where(sel_mask > 0., coordinates, tf.zeros_like(coordinates) - self.masked_coordinate_offset)

        collected_neighbours = self.collect_neighbours(coordinates, x, mask)
        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)
        
        if self.masked_coordinate_offset is not None:
            output *= mask

        if self.also_coordinates:
            return [output, coordinates]
        return output

    def compute_output_shape(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        if self.also_coordinates:
            return [(input_shape[0], input_shape[1], self.output_feature_transform.units),
                    (input_shape[0], input_shape[1], self.n_dimensions)]
        
        return (input_shape[0], input_shape[1], self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, x, mask):
        distance_matrix = euclidean_squared(coordinates, coordinates)
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix, self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]
        features = self.input_feature_transform(x)
        n_batches = tf.shape(features)[0]
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]

        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)  # (B, 1, 1, 1)

        batch_indices = tf.tile(batch_range, [1, n_vertices, self.n_neighbours - 1, 1])  # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices, axis=3)  # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)
    
        distance = -ranked_distances[:, :, 1:]
        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)

        for i in range(len(self.message_passing_layers) + 1):
            if i:
                features = self.message_passing_layers[i-1](tf.concat([features, x], axis=-1))
                w = self.message_parsing_distance_weights[i-1]
                weights = gauss_of_lin(w * distance)
                weights = tf.expand_dims(weights, axis=-1)

            if 0 < self.feature_dropout < 1:
                features = tf.keras.layers.Dropout(self.feature_dropout)(features)

            neighbour_features = tf.gather_nd(features, indices)  # (B, V, N-1, F)
            neighbour_features *= weights

            neighbours_max = tf.reduce_max(neighbour_features, axis=2)
            neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)

            features = tf.concat([neighbours_max, neighbours_mean], axis=-1)
            if mask is not None:
                features *= mask
        
        return features

    def get_config(self):
        config = {
            'n_neighbours': self.n_neighbours, 
            'n_dimensions': self.n_dimensions, 
            'n_filters': self.n_filters, 
            'n_propagate': self.n_propagate,
            'subname': self.subname,
            'also_coordinates': self.also_coordinates,
            'feature_dropout': self.feature_dropout,
            'masked_coordinate_offset': self.masked_coordinate_offset,
            'additional_message_passing': self.additional_message_passing
        }
        base_config = super(GravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

global_layers_list['GravNet_simple'] = GravNet_simple   
global_layers_list['GravNet'] = GravNet