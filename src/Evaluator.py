import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import random 

class Evaluator:
    
    def __init__(self, X=None, y=None, misc=None, is_intersections=False):
        self.is_intersections = is_intersections
        self.X = tf.convert_to_tensor(X) if X is not None else tf.constant(0)
        self.y = tf.convert_to_tensor(y) if y is not None else tf.constant(0)
        self.misc = tf.convert_to_tensor(misc) if misc is not None else tf.constant(0)
        self.dataframe = self._create_dataframe_structure()
        self.tD = 0
        self.tB = 0
        
        
        
    def _create_dataframe_structure(self):
        if self.is_intersections:
            columns = [
                'event', 'energy_A','energy_B','energy_C',
                'time_A', 'time_B','time_C',
                'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
                'sector', 'layer', 'rec_pid', 'pindex', 'mc_pid',
                'unique_otid', 'beta', 'xc', 'yc', 'cluster_id', 'is_cluster_leader'
            ]
        else:
            columns = [
                'event', 'energy', 'time', 'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
                'sector', 'layer', 'centroid_x', 'centroid_y', 'rec_pid', 'pindex', 'mc_pid',
                'unique_otid', 'beta', 'xc', 'yc', 'cluster_id', 'is_cluster_leader', 'pred_centroid_x','pred_centroid_y'
            ]
        
        return pd.DataFrame(columns=columns)
    
    
    
    def get_event_dataframe(self, event_number=None):
        if event_number is None:
            event_number = random.choice(self.dataframe['event'].unique())
            print("Randomly generated event number =",event_number)
        return self.dataframe[self.dataframe['event'] == event_number]
    
    
    
    def load_model(self, model):
        self.model = model
        
        
        
    def predict(self):
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Error: Must load a model before predicting.")
        
        out = self.model.predict(self.X)  # out is N x M x 3
        
        N, M, _ = self.X.shape
        
        try:
#             # Extract sector and layer from one-hot encoded columns
#             if self.is_intersections == False:
#                 sector = np.argmax(self.X.numpy()[:,:,8:14], axis=2) + 1
#                 layer = np.argmax(self.X.numpy()[:,:,14:23], axis=2) + 1
#             else:
#                 sector = np.argmax(self.X.numpy()[:,:,11:17], axis=2) + 1
#                 layer = np.argmax(self.X.numpy()[:,:,8:11], axis=2) + 1

            if self.is_intersections == False:
                df_data = {
                    'event': np.repeat(np.arange(N), M),
                    'energy': self.X.numpy()[:,:,0].flatten(),
                    'time': self.X.numpy()[:,:,1].flatten(),
                    'xo': self.X.numpy()[:,:,2].flatten(),
                    'yo': self.X.numpy()[:,:,3].flatten(),
                    'zo': self.X.numpy()[:,:,4].flatten(),
                    'xe': self.X.numpy()[:,:,5].flatten(),
                    'ye': self.X.numpy()[:,:,6].flatten(),
                    'ze': self.X.numpy()[:,:,7].flatten(),
                    #'sector': sector.flatten(),
                    #'layer': layer.flatten(),
                    'centroid_x': self.X.numpy()[:,:,17].flatten(),
                    'centroid_y': self.X.numpy()[:,:,18].flatten(),
                    'rec_pid': self.misc.numpy()[:,:,0].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'pindex': self.misc.numpy()[:,:,1].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'mc_pid': self.misc.numpy()[:,:,2].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'unique_otid': self.y.numpy()[:,:,0].flatten(),
                    'beta': out[:,:,0].flatten(),
                    'xc': out[:,:,1].flatten(),
                    'yc': out[:,:,2].flatten(),
                    'pred_centroid_x': out[:,:,3].flatten(),
                    'pred_centroid_y': out[:,:,4].flatten(),
                    'cluster_id': -1,
                    'is_cluster_leader': 0
                }
            else:
                df_data = {
                    'event': np.repeat(np.arange(N), M),
                    'xo': self.X.numpy()[:,:,0].flatten(),
                    'yo': self.X.numpy()[:,:,1].flatten(),
                    'xe': self.X.numpy()[:,:,0].flatten(),
                    'ye': self.X.numpy()[:,:,1].flatten(),
                    'zo': 0*self.X.numpy()[:,:,0].flatten(),
                    'ze': 0*self.X.numpy()[:,:,1].flatten(),
                    'energy_A': self.X.numpy()[:,:,2].flatten(),
                    'energy_B': self.X.numpy()[:,:,3].flatten(),
                    'energy_C': self.X.numpy()[:,:,4].flatten(),
                    'time_A': self.X.numpy()[:,:,5].flatten(),
                    'time_B': self.X.numpy()[:,:,6].flatten(),
                    'time_C': self.X.numpy()[:,:,7].flatten(),
                    #'sector': sector.flatten(),
                    #'layer': layer.flatten(),
                    'rec_pid': self.misc.numpy()[:,:,0].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'pindex': self.misc.numpy()[:,:,1].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'mc_pid': self.misc.numpy()[:,:,2].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'unique_otid': self.y.numpy()[:,:,0].flatten(),
                    'beta': out[:,:,0].flatten(),
                    'xc': out[:,:,1].flatten(),
                    'yc': out[:,:,2].flatten(),
                    'cluster_id': -1,
                    'is_cluster_leader': 0
                }
                
        except:
            if self.is_intersections == False:
                raise ValueError("NOT IMPLEMENTED")
            else:
                df_data = {
                    'event': np.repeat(np.arange(N), M),
                    'xo': self.X.numpy()[:,:,0].flatten(),
                    'yo': self.X.numpy()[:,:,1].flatten(),
                    'xe': self.X.numpy()[:,:,0].flatten(),
                    'ye': self.X.numpy()[:,:,1].flatten(),
                    'zo': 0*self.X.numpy()[:,:,0].flatten(),
                    'ze': 0*self.X.numpy()[:,:,1].flatten(),
                    'rec_pid': self.misc.numpy()[:,:,0].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'pindex': self.misc.numpy()[:,:,1].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'mc_pid': self.misc.numpy()[:,:,2].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
                    'unique_otid': self.y.numpy()[:,:,0].flatten(),
                    'beta': out[:,:,0].flatten(),
                    'xc': out[:,:,1].flatten(),
                    'yc': out[:,:,2].flatten(),
                    'cluster_id': -1,
                    'is_cluster_leader': 0
                }
        self.dataframe = pd.DataFrame(df_data)

        # Remove rows where all specified columns are zero
        columns_to_check = ['xo', 'yo', 'zo', 'xe', 'ye', 'ze']
        mask = self.dataframe[columns_to_check].eq(0).all(axis=1)
        self.dataframe = self.dataframe[~mask]
    
    
    
    def cluster(self, tB, tD):
        if not 0 < tB <= 1:
            raise ValueError("Error: tB must be between 0 and 1.")
        if tD <= 0:
            raise ValueError("Error: tD must be a positive value.")
        
        self.tB = tB
        self.tD = tD
        
        for event_id in self.dataframe['event'].unique():
            event_data = self.dataframe[self.dataframe['event'] == event_id]
            event_data_sorted = event_data.sort_values(by='beta', ascending=False)
            
            cluster_id = 0
            while not event_data_sorted.empty:
                # Get the highest beta row (leader)
                leader = event_data_sorted.iloc[0]
                if leader['beta'] <= tB:
                    break
                
                # Assign cluster_id to the leader
                self.dataframe.loc[leader.name, 'cluster_id'] = cluster_id
                self.dataframe.loc[leader.name, 'is_cluster_leader'] = 1
                
                # Find all points within the distance tD from the leader
                remaining_points = event_data_sorted.iloc[1:]
                distances = cdist([(leader['xc'], leader['yc'])], remaining_points[['xc', 'yc']].values)[0]
                close_points = remaining_points[distances <= tD]
                
                # Assign cluster_id to close points
                self.dataframe.loc[close_points.index, 'cluster_id'] = cluster_id
                event_data_sorted = event_data_sorted.drop(close_points.index.union([leader.name]))
                
                # Increment cluster_id for the next cluster
                cluster_id += 1
