import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import random 

class Evaluator:
    
    def __init__(self, X=None, y=None, misc=None):
        self.X = tf.convert_to_tensor(X) if X is not None else tf.constant(0)
        self.y = tf.convert_to_tensor(y) if y is not None else tf.constant(0)
        self.misc = tf.convert_to_tensor(misc) if misc is not None else tf.constant(0)
        self.dataframe = self._create_dataframe_structure()
        self.tD = 0
        self.tB = 0
        
        
        
    def _create_dataframe_structure(self):
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
        
        sector = np.argmax(self.X.numpy()[:,:,23:29], axis=2) + 1
        layer = np.argmax(self.X.numpy()[:,:,8:17], axis=2) + 1

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
            'sector': sector.flatten(),
            'layer': layer.flatten(),
            'centroid_x': self.X.numpy()[:,:,17].flatten(),
            'centroid_y': self.X.numpy()[:,:,18].flatten(),
            'is_3way_same_group': self.X.numpy()[:,:,19].flatten(),
            'is_2way_same_group': self.X.numpy()[:,:,20].flatten(),
            'is_3way_cross_group': self.X.numpy()[:,:,21].flatten(),
            'is_2way_cross_group': self.X.numpy()[:,:,22].flatten(),
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
                
    def calculate_calorimeter_clusters(self, option):
        if option not in ["with_clustering", "with_recon", "with_truth"]:
            raise ValueError("Option must be 'with_clustering', 'with_recon', or 'with_truth'.")
        
        indexing_var = {'with_clustering': 'cluster_id', 'with_recon': 'pindex', 'with_truth': 'unique_otid'}[option]
        clusterX_col = f"calorimeter_clusterX_{option}"
        clusterY_col = f"calorimeter_clusterY_{option}"
        
        self.dataframe[clusterX_col] = 0.0
        self.dataframe[clusterY_col] = 0.0
        
        for event_id, event_data in self.dataframe.groupby('event'):
            for idx_value, group in event_data.groupby(indexing_var):
                if idx_value == -1:
                    continue
                
                sector_counts = group['sector'].value_counts()
                most_common_sector = sector_counts.idxmax()
                
                if sector_counts[most_common_sector] == 1:
                    continue
                
                priority_columns = [
                    'is_3way_same_group',
                    'is_2way_same_group',
                    'is_3way_cross_group',
                    'is_2way_cross_group'
                ]
                
                found = False
                for col in priority_columns:
                    filtered_group = group[(group['sector'] == most_common_sector) & (group[col] == 1)]
                    if not filtered_group.empty:
                        mean_centroid_x = filtered_group['centroid_x'].mean()
                        mean_centroid_y = filtered_group['centroid_y'].mean()
                        self.dataframe.loc[group.index, clusterX_col] = mean_centroid_x
                        self.dataframe.loc[group.index, clusterY_col] = mean_centroid_y
                        found = True
                        break
                
                if not found:
                    self.dataframe.loc[group.index, clusterX_col] = 0.0
                    self.dataframe.loc[group.index, clusterY_col] = 0.0