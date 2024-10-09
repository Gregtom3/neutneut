import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import random 
from TrainData import load_unzip_data
from global_params import *
from ECALClusterAnalyzer import ECALClusterAnalyzer
import shutil
import hipopy as hp  
from loss_functions import condensation_loss

class Evaluator:
    
    def __init__(self, h5_filename=None, X=None, y=None, misc=None, Nevents=None, original_hipofile=None, save_cluster=False):
        self.h5_filename = h5_filename
        if self.h5_filename is not None:
            # Load the data if h5_filename is provided
            (X, y, misc) = load_unzip_data(h5_filename)

        # If Nevents is provided, limit the data to Nevents rows
        if Nevents is not None and X is not None:
            X = X[:Nevents]
            y = y[:Nevents]
            misc = misc[:Nevents]

        # Convert to TensorFlow tensors or fallback to constant 0
        self.X = tf.convert_to_tensor(X) if X is not None else tf.constant(0)
        self.y = tf.convert_to_tensor(y) if y is not None else tf.constant(0)
        self.misc = tf.convert_to_tensor(misc) if misc is not None else tf.constant(0)

        # Create the dataframe structure
        self.dataframe = self._create_dataframe_structure()
        self.tD = 0
        self.tB = 0

        if original_hipofile!=None:
            self.original_hipofile = original_hipofile
            self.new_hipofile = self.original_hipofile.replace(".hipo","_OC.hipo")
        self.clusters_df = None
        self.save_cluster = save_cluster
        
    @classmethod
    def from_data(cls, X, y, misc, Nevents=None):
        # Initialize the Evaluator by directly passing data
        return cls(X=X, y=y, misc=misc, Nevents=Nevents)
        
        
    def _create_dataframe_structure(self):
        columns = [
            'event', 'energy', 'time', 'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
            'sector', 'layer', 'centroid_x', 'centroid_y', 'centroid_z', 'rec_pid', 'pindex', 'mc_pid', 'file_event',
            'unique_otid', 'beta', 'xc', 'yc', 'cluster_id', 'is_cluster_leader', 'pred_centroid_x','pred_centroid_y','pred_pid'
        ]
        
        return pd.DataFrame(columns=columns)
    
    
    
    def get_event_dataframe(self, event=None):
        if event is None:
            event = random.choice(self.dataframe['event'].unique())
            print("Randomly generated event number =",event)
        event = self.dataframe['event'].unique()[event] # Event is global now
        return self.dataframe[self.dataframe['event'] == event]
    
    
    
    def load_model(self, model):
        self.model = model
        
        
        
    def predict(self):
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Error: Must load a model before predicting.")

        out = self.model.predict(self.X)  # out is N x M x 3
        
        N, M, _ = self.X.shape
        
        sector = np.argmax(self.X.numpy()[:,:,22:28], axis=2) + 1
        layer = np.argmax(self.X.numpy()[:,:,8:17], axis=2) + 1

        df_data = {
            'event': self.misc.numpy()[:,:,3].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
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
            'centroid_z': self.X.numpy()[:,:,19].flatten(),
            'is_3way_same_group': self.X.numpy()[:,:,20].flatten(),
            'is_2way_same_group': self.X.numpy()[:,:,21].flatten(),
            'rec_pid': self.misc.numpy()[:,:,0].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
            'pindex': self.misc.numpy()[:,:,1].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
            'mc_pid': self.misc.numpy()[:,:,2].flatten() if not tf.reduce_all(tf.equal(self.misc, 0)) else np.zeros(N * M),
            'unique_otid': self.y.numpy()[:,:,0].flatten(),
            'beta': out[:,:,0].flatten(),
            'xc': out[:,:,1].flatten(),
            'yc': out[:,:,2].flatten(),
            'cluster_id': -1,
            'is_cluster_leader': 0,
            'pred_pid': np.argmax(out[:,:,3:6],axis=2).flatten(),
            'pred_photon': out[:,:,3].flatten(),
            'pred_neutron': out[:,:,4].flatten(),
            'pred_other': out[:,:,5].flatten()
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
        
        for event_id in tqdm(self.dataframe['event'].unique()):
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

        self.create_cluster_dataframe()
        self.match_clusters()
        
        if self.save_cluster:
            self.dataframe.to_csv(self.h5_filename.replace("/training/","/predict/"))
    
    def get_loss_df(self,q_min=0.1):
        events = self.dataframe.event.unique()
        # Initialize an empty list to store the individual event DataFrames
        df_list = []

        for event in events:
            # Get the dataframe for a specific event
            ev_df = self.dataframe[self.dataframe.event == event]
            n_objs = len(ev_df.unique_otid.unique())  # Number of unique objects
            n_hits = len(ev_df)  # Number of hits

            # Convert data to tensors
            unique_otid = tf.convert_to_tensor(ev_df.unique_otid.values, dtype=tf.float32)
            latent_coords = tf.convert_to_tensor(ev_df[["xc", "yc"]].values, dtype=tf.float32)
            beta = tf.convert_to_tensor(ev_df.beta.values, dtype=tf.float32)
            beta = tf.reshape(beta, [-1, 1])  # Reshaping to [N, 1]

            # Noise threshold (you can adjust the noise value if needed)
            noise = -1

            # Calculate the losses using the condensation_loss function
            loss_dict = condensation_loss(q_min=q_min,
                                          object_id=np.array(unique_otid),
                                          beta=beta,
                                          x=latent_coords,
                                          noise_threshold=noise)

            # Extract individual losses
            att_loss = loss_dict['attractive']
            rep_loss = loss_dict['repulsive']
            cow_loss = loss_dict['coward']
            nse_loss = loss_dict['noise']
            tot_loss = att_loss + rep_loss + cow_loss + nse_loss

            # Create a small DataFrame for this event
            event_df = pd.DataFrame([{
                'event': event,
                'n_objs': n_objs,
                'n_hits': n_hits,
                'att_loss': att_loss.numpy(),  # Convert tensor to numpy for storing in the dataframe
                'rep_loss': rep_loss.numpy(),
                'cow_loss': cow_loss.numpy(),
                'nse_loss': nse_loss.numpy(),
                'tot_loss': tot_loss.numpy()
            }])

            # Add this event's DataFrame to the list
            df_list.append(event_df)

        # Concatenate all small DataFrames into the final DataFrame
        loss_df = pd.concat(df_list, ignore_index=True)

        return loss_df
            
    def create_cluster_dataframe(self):
        """
        Converts the Evaluator dataframe into the desired structure with unscaled values.

        Parameters:
        -----------
        evaluator : Evaluator
            The instance of the Evaluator class.

        Returns:
        --------
        pd.DataFrame
            A new DataFrame with unscaled values for specified columns.
        """
        df = self.dataframe.copy()
        
        # df = df[df['is_cluster_leader']==1]
        # Undo scaling for Group 1: ['xo', 'yo', 'xe', 'ye', 'centroid_x', 'centroid_y']
        group_1 = ['xo', 'yo', 'xe', 'ye', 'centroid_x', 'centroid_y']
        df[group_1] = ECAL_xy_min - ECAL_xy_min * df[group_1] + ECAL_xy_max * df[group_1]

        # Undo scaling for Group 2: ['zo', 'ze', 'centroid_z']
        group_2 = ['zo', 'ze', 'centroid_z']
        df[group_2] = ECAL_z_min - ECAL_z_min * df[group_2] + ECAL_z_max * df[group_2]

        # Undo scaling for Group 3: ['energy']
        group_3 = ['energy']
        df[group_3] = ECAL_energy_min - ECAL_energy_min * df[group_3] + ECAL_energy_max * df[group_3]

        # Undo scaling for Group 4: ['time']
        group_4 = ['time']
        df[group_4] = ECAL_time_min - ECAL_time_min * df[group_4] + ECAL_time_max * df[group_4]
        
        # Set 'id' column to 0
        df['id'] = 0
        
        self.dataframe = df
        cluster_ana = ECALClusterAnalyzer(self.dataframe, clustering_variable="cluster_id")
        self.clusters_df = cluster_ana.create_clusters()
        
    def match_clusters(self):

        df_hits = self.dataframe
        df_clusters = self.clusters_df
        # Create new columns for cluster_x, cluster_y, cluster_z initialized with NaN
        df_hits['cluster_x'] = np.nan
        df_hits['cluster_y'] = np.nan
        df_hits['cluster_z'] = np.nan

        # Iterate through each row in df_hits
        for index, row in df_hits.iterrows():
            unique_otid = row['cluster_id']
            layer = row['layer']
            event = row['event']  # Ensure we're matching the same event

            # Find the matching rows in df_clusters based on UID == unique_otid and the same event
            try:
                matching_clusters = df_clusters[(df_clusters['uid'] == unique_otid) & (df_clusters['event'] == event)]
            except:
                return
            # Determine which layer to match (1, 4, or 7 based on the layer range)
            if 1 <= layer <= 3:
                matching_cluster = matching_clusters[matching_clusters['layer'] == 1]
            elif 4 <= layer <= 6:
                matching_cluster = matching_clusters[matching_clusters['layer'] == 4]
            elif 7 <= layer <= 9:
                matching_cluster = matching_clusters[matching_clusters['layer'] == 7]
            else:
                matching_cluster = None

            # If there's a match, assign the x, y, z values to df_hits, otherwise leave NaN
            if matching_cluster is not None and not matching_cluster.empty:
                df_hits.at[index, 'cluster_x'] = matching_cluster.iloc[0]['x']
                df_hits.at[index, 'cluster_y'] = matching_cluster.iloc[0]['y']
                df_hits.at[index, 'cluster_z'] = matching_cluster.iloc[0]['z']

        self.dataframe = df_hits
    
    def write_hipo_file(self):
            """
            Write the cluster data into a new hipo file using the generated clusters DataFrame.
            """

            # Copy new_hipofile
            shutil.copy2(self.original_hipofile, self.new_hipofile)
            print("***** NEW HIPO FILE = ",self.new_hipofile," *********")

            # Define the bank names and types for ECAL::clusters_OC
            cluster_bank = "ECAL::clusters_OC"
            cluster_names = ["id", "status", "sector", "layer", "x", "y", "z", "energy", "time", 
                             "widthU", "widthV", "widthW", "idU", "idV", "idW", "coordU", "coordV", "coordW"]
            cluster_names_and_types = {
                "id": "S", "status": "S", "sector": "B", "layer": "B", "x": "F", "y": "F", "z": "F", 
                "energy": "F", "time": "F", "widthU": "F", "widthV": "F", "widthW": "F", 
                "idU": "B", "idV": "B", "idW": "B", "coordU": "I", "coordV": "I", "coordW": "I"
            }

            # Define the bank names and types for ECAL::moments_OC
            moments_bank = "ECAL::moments_OC"
            moments_names = ["distU", "distV", "distW", "m1u", "m1v", "m1w", "m2u", "m2v", "m2w", "m3u", "m3v", "m3w"]
            moments_names_and_types = {
                "distU": "F", "distV": "F", "distW": "F", "m1u": "F", "m1v": "F", "m1w": "F", 
                "m2u": "F", "m2v": "F", "m2w": "F", "m3u": "F", "m3v": "F", "m3w": "F"
            }

            # Define the bank names and types for ECAL::calib_OC
            calib_bank = "ECAL::calib_OC"
            calib_names = ["sector", "layer", "size", "dbstU", "dbstV", "dbstW", "rawEU", "rawEV", "rawEW", 
                           "recEU", "recEV", "recEW", "recDTU", "recDTV", "recDTW", "recFTU", "recFTV", "recFTW"]
            calib_names_and_types = {
                "sector": "B", "layer": "B", "size": "F", "dbstU": "S", "dbstV": "S", "dbstW": "S", 
                "rawEU": "F", "rawEV": "F", "rawEW": "F", "recEU": "F", "recEV": "F", "recEW": "F",
                "recDTU": "F", "recDTV": "F", "recDTW": "F", "recFTU": "F", "recFTV": "F", "recFTW": "F"
            }

            # Open the new hipo file
            file = hp.recreate(self.new_hipofile)
            file.newTree(cluster_bank, cluster_names_and_types)
            file.open()

            # Iterate through events and write data to the hipo file
            for event, _ in enumerate(file):
                event_group = self.clusters_df[self.clusters_df['event'] == event]
                # Cluster data
                cluster_data = [
                    [int(uid) for uid in event_group["uid"].tolist()],            # 'id' should be short ('S')
                    [int(status) for status in event_group["status"].tolist()],   # 'status' should be short ('S')
                    [int(sector) for sector in event_group["sector"].tolist()],   # 'sector' should be byte ('B')
                    [int(layer) for layer in event_group["layer"].tolist()],      # 'layer' should be byte ('B')
                    [float(x) for x in event_group["x"].tolist()],                # 'x' should be float ('F')
                    [float(y) for y in event_group["y"].tolist()],                # 'y' should be float ('F')
                    [float(z) for z in event_group["z"].tolist()],                # 'z' should be float ('F')
                    [float(energy) for energy in event_group["energy"].tolist()], # 'energy' should be float ('F')
                    [float(time) for time in event_group["time"].tolist()],       # 'time' should be float ('F')
                    [float(widthU) for widthU in event_group["widthU"].tolist()], # 'widthU' should be float ('F')
                    [float(widthV) for widthV in event_group["widthV"].tolist()], # 'widthV' should be float ('F')
                    [float(widthW) for widthW in event_group["widthW"].tolist()], # 'widthW' should be float ('F')
                    [int(idU) for idU in event_group["idU"].tolist()],            # 'idU' should be byte ('B')
                    [int(idV) for idV in event_group["idV"].tolist()],            # 'idV' should be byte ('B')
                    [int(idW) for idW in event_group["idW"].tolist()],            # 'idW' should be byte ('B')
                    [int(coordU) for coordU in event_group["coordU"].tolist()],   # 'coordU' should be integer ('I')
                    [int(coordV) for coordV in event_group["coordV"].tolist()],   # 'coordV' should be integer ('I')
                    [int(coordW) for coordW in event_group["coordW"].tolist()]    # 'coordW' should be integer ('I')
                ]
                # Write data for the event to the clusters bank
                file.update({cluster_bank: cluster_data})

            # Close the hipo file
            file.close() 

            file = hp.recreate(self.new_hipofile)
            file.newTree(moments_bank, moments_names_and_types)
            file.open()
            # Iterate through events and write data to the hipo file
            for event, _ in enumerate(file):
                event_group = self.clusters_df[self.clusters_df['event'] == event]

                # Moments data, all zeros, with the same length as the cluster data entries
                num_entries = len(event_group["uid"])
                moments_data = [
                    [0.0] * num_entries,  # 'distU' should be float ('F')
                    [0.0] * num_entries,  # 'distV' should be float ('F')
                    [0.0] * num_entries,  # 'distW' should be float ('F')
                    [0.0] * num_entries,  # 'm1u' should be float ('F')
                    [0.0] * num_entries,  # 'm1v' should be float ('F')
                    [0.0] * num_entries,  # 'm1w' should be float ('F')
                    [0.0] * num_entries,  # 'm2u' should be float ('F')
                    [0.0] * num_entries,  # 'm2v' should be float ('F')
                    [0.0] * num_entries,  # 'm2w' should be float ('F')
                    [0.0] * num_entries,  # 'm3u' should be float ('F')
                    [0.0] * num_entries,  # 'm3v' should be float ('F')
                    [0.0] * num_entries   # 'm3w' should be float ('F')
                ]

                # Write data for the event to the moments bank
                file.update({moments_bank: moments_data})

            # Close the hipo file
            file.close()

            file = hp.recreate(self.new_hipofile)
            file.newTree(calib_bank, calib_names_and_types)
            file.open()
            # Iterate through events and write data to the hipo file
            for event, _ in enumerate(file):
                event_group = self.clusters_df[self.clusters_df['event'] == event]

                # Calib data, all zeros, same length as the cluster data
                num_entries = len(event_group["uid"])
                calib_data = [
                    [0] * num_entries,  # 'sector' as byte
                    [0] * num_entries,  # 'layer' as byte
                    [0.0] * num_entries,  # 'size' as float
                    [0] * num_entries,  # 'dbstU' as short
                    [0] * num_entries,  # 'dbstV' as short
                    [0] * num_entries,  # 'dbstW' as short
                    [0.0] * num_entries,  # 'rawEU' as float
                    [0.0] * num_entries,  # 'rawEV' as float
                    [0.0] * num_entries,  # 'rawEW' as float
                    [0.0] * num_entries,  # 'recEU' as float
                    [0.0] * num_entries,  # 'recEV' as float
                    [0.0] * num_entries,  # 'recEW' as float
                    [0.0] * num_entries,  # 'recDTU' as float
                    [0.0] * num_entries,  # 'recDTV' as float
                    [0.0] * num_entries,  # 'recDTW' as float
                    [0.0] * num_entries,  # 'recFTU' as float
                    [0.0] * num_entries,  # 'recFTV' as float
                    [0.0] * num_entries   # 'recFTW' as float
                ]

                # Write data for the event to the calib bank
                file.update({calib_bank: calib_data})

            # Close the hipo file
            file.close()





    # def write_hipo_file(self):
    #         """
    #         Write the cluster data into a new hipo file using the generated clusters DataFrame.
    #         """

    #         # Copy new_hipofile
    #         shutil.copy2(self.original_hipofile, self.new_hipofile)
    #         print("***** NEW HIPO FILE = ",self.new_hipofile," *********")

    #         # Define the bank names and types for ECAL::clusters_OC
    #         cluster_bank = "ECAL::clusters_OC"
    #         cluster_names = ["id", "status", "sector", "layer", "x", "y", "z", "energy", "time", 
    #                          "widthU", "widthV", "widthW", "idU", "idV", "idW", "coordU", "coordV", "coordW"]
    #         cluster_names_and_types = {
    #             "id": "S", "status": "S", "sector": "B", "layer": "B", "x": "F", "y": "F", "z": "F", 
    #             "energy": "F", "time": "F", "widthU": "F", "widthV": "F", "widthW": "F", 
    #             "idU": "B", "idV": "B", "idW": "B", "coordU": "I", "coordV": "I", "coordW": "I"
    #         }

    #         # Define the bank names and types for ECAL::moments_OC
    #         moments_bank = "ECAL::moments_OC"
    #         moments_names = ["distU", "distV", "distW", "m1u", "m1v", "m1w", "m2u", "m2v", "m2w", "m3u", "m3v", "m3w"]
    #         moments_names_and_types = {
    #             "distU": "F", "distV": "F", "distW": "F", "m1u": "F", "m1v": "F", "m1w": "F", 
    #             "m2u": "F", "m2v": "F", "m2w": "F", "m3u": "F", "m3v": "F", "m3w": "F"
    #         }

    #         # Define the bank names and types for ECAL::calib_OC
    #         calib_bank = "ECAL::calib_OC"
    #         calib_names = ["sector", "layer", "size", "dbstU", "dbstV", "dbstW", "rawEU", "rawEV", "rawEW", 
    #                        "recEU", "recEV", "recEW", "recDTU", "recDTV", "recDTW", "recFTU", "recFTV", "recFTW"]
    #         calib_names_and_types = {
    #             "sector": "B", "layer": "B", "size": "F", "dbstU": "S", "dbstV": "S", "dbstW": "S", 
    #             "rawEU": "F", "rawEV": "F", "rawEW": "F", "recEU": "F", "recEV": "F", "recEW": "F",
    #             "recDTU": "F", "recDTV": "F", "recDTW": "F", "recFTU": "F", "recFTV": "F", "recFTW": "F"
    #         }

    #         # Open the new hipo file
    #         file = hp.recreate(self.new_hipofile)
    #         file.newTree(cluster_bank, cluster_names_and_types)
    #         file.open()

    #         # Iterate through events and write data to the hipo file
    #         for event, _ in enumerate(file):
    #             event_group = self.clusters_df[self.clusters_df['event'] == event]
            
    #             # Check if the event_group is empty
    #             if event_group.empty:
    #                 # Create a single entry with all zeros when cluster data is empty
    #                 cluster_data = [
    #                     [0],          # 'id' should be short ('S')
    #                     [0],          # 'status' should be short ('S')
    #                     [0],          # 'sector' should be byte ('B')
    #                     [0],          # 'layer' should be byte ('B')
    #                     [0.0],        # 'x' should be float ('F')
    #                     [0.0],        # 'y' should be float ('F')
    #                     [0.0],        # 'z' should be float ('F')
    #                     [0.0],        # 'energy' should be float ('F')
    #                     [0.0],        # 'time' should be float ('F')
    #                     [0.0],        # 'widthU' should be float ('F')
    #                     [0.0],        # 'widthV' should be float ('F')
    #                     [0.0],        # 'widthW' should be float ('F')
    #                     [0],          # 'idU' should be byte ('B')
    #                     [0],          # 'idV' should be byte ('B')
    #                     [0],          # 'idW' should be byte ('B')
    #                     [0],          # 'coordU' should be integer ('I')
    #                     [0],          # 'coordV' should be integer ('I')
    #                     [0]           # 'coordW' should be integer ('I')
    #                 ]
    #             else:
    #                 # Cluster data
    #                 cluster_data = [
    #                     [int(uid) for uid in event_group["uid"].tolist()],            # 'id' should be short ('S')
    #                     [int(status) for status in event_group["status"].tolist()],   # 'status' should be short ('S')
    #                     [int(sector) for sector in event_group["sector"].tolist()],   # 'sector' should be byte ('B')
    #                     [int(layer) for layer in event_group["layer"].tolist()],      # 'layer' should be byte ('B')
    #                     [float(x) for x in event_group["x"].tolist()],                # 'x' should be float ('F')
    #                     [float(y) for y in event_group["y"].tolist()],                # 'y' should be float ('F')
    #                     [float(z) for z in event_group["z"].tolist()],                # 'z' should be float ('F')
    #                     [float(energy) for energy in event_group["energy"].tolist()], # 'energy' should be float ('F')
    #                     [float(time) for time in event_group["time"].tolist()],       # 'time' should be float ('F')
    #                     [float(widthU) for widthU in event_group["widthU"].tolist()], # 'widthU' should be float ('F')
    #                     [float(widthV) for widthV in event_group["widthV"].tolist()], # 'widthV' should be float ('F')
    #                     [float(widthW) for widthW in event_group["widthW"].tolist()], # 'widthW' should be float ('F')
    #                     [int(idU) for idU in event_group["idU"].tolist()],            # 'idU' should be byte ('B')
    #                     [int(idV) for idV in event_group["idV"].tolist()],            # 'idV' should be byte ('B')
    #                     [int(idW) for idW in event_group["idW"].tolist()],            # 'idW' should be byte ('B')
    #                     [int(coordU) for coordU in event_group["coordU"].tolist()],   # 'coordU' should be integer ('I')
    #                     [int(coordV) for coordV in event_group["coordV"].tolist()],   # 'coordV' should be integer ('I')
    #                     [int(coordW) for coordW in event_group["coordW"].tolist()]    # 'coordW' should be integer ('I')
    #                 ]
            
    #             # Write data for the event to the clusters bank
    #             file.update({cluster_bank: np.array(cluster_data)})
    #         # Close the hipo file
    #         file.close() 

    #         file = hp.recreate(self.new_hipofile)
    #         file.newTree(moments_bank, moments_names_and_types)
    #         file.open()
    #         # Iterate through events and write data to the hipo file
    #         for event, _ in enumerate(file):
    #             event_group = self.clusters_df[self.clusters_df['event'] == event]
                
    #             # Check if the event_group is empty
    #             if event_group.empty:
    #                 # Moments data with all zeros for a single entry when cluster data is empty
    #                 moments_data = [
    #                     [0.0],  # 'distU' should be float ('F')
    #                     [0.0],  # 'distV' should be float ('F')
    #                     [0.0],  # 'distW' should be float ('F')
    #                     [0.0],  # 'm1u' should be float ('F')
    #                     [0.0],  # 'm1v' should be float ('F')
    #                     [0.0],  # 'm1w' should be float ('F')
    #                     [0.0],  # 'm2u' should be float ('F')
    #                     [0.0],  # 'm2v' should be float ('F')
    #                     [0.0],  # 'm2w' should be float ('F')
    #                     [0.0],  # 'm3u' should be float ('F')
    #                     [0.0],  # 'm3v' should be float ('F')
    #                     [0.0]   # 'm3w' should be float ('F')
    #                 ]
    #             else:
    #                 # Moments data, all zeros, with the same length as the cluster data entries
    #                 num_entries = len(event_group["uid"])
    #                 moments_data = [
    #                     [0.0] * num_entries,  # 'distU' should be float ('F')
    #                     [0.0] * num_entries,  # 'distV' should be float ('F')
    #                     [0.0] * num_entries,  # 'distW' should be float ('F')
    #                     [0.0] * num_entries,  # 'm1u' should be float ('F')
    #                     [0.0] * num_entries,  # 'm1v' should be float ('F')
    #                     [0.0] * num_entries,  # 'm1w' should be float ('F')
    #                     [0.0] * num_entries,  # 'm2u' should be float ('F')
    #                     [0.0] * num_entries,  # 'm2v' should be float ('F')
    #                     [0.0] * num_entries,  # 'm2w' should be float ('F')
    #                     [0.0] * num_entries,  # 'm3u' should be float ('F')
    #                     [0.0] * num_entries,  # 'm3v' should be float ('F')
    #                     [0.0] * num_entries   # 'm3w' should be float ('F')
    #                 ]
            
    #             # Write data for the event to the moments bank
    #             file.update({moments_bank: np.array(moments_data)})
    #         # Close the hipo file
    #         file.close()

    #         file = hp.recreate(self.new_hipofile)
    #         file.newTree(calib_bank, calib_names_and_types)
    #         file.open()
    #         # Iterate through events and write data to the hipo file
    #         for event, _ in enumerate(file):
    #             event_group = self.clusters_df[self.clusters_df['event'] == event]
            
    #             # Check if the event_group is empty
    #             if event_group.empty:
    #                 # Calib data with all zeros for a single entry when cluster data is empty
    #                 calib_data = [
    #                     [0],        # 'sector' as byte
    #                     [0],        # 'layer' as byte
    #                     [0.0],      # 'size' as float
    #                     [0],        # 'dbstU' as short
    #                     [0],        # 'dbstV' as short
    #                     [0],        # 'dbstW' as short
    #                     [0.0],      # 'rawEU' as float
    #                     [0.0],      # 'rawEV' as float
    #                     [0.0],      # 'rawEW' as float
    #                     [0.0],      # 'recEU' as float
    #                     [0.0],      # 'recEV' as float
    #                     [0.0],      # 'recEW' as float
    #                     [0.0],      # 'recDTU' as float
    #                     [0.0],      # 'recDTV' as float
    #                     [0.0],      # 'recDTW' as float
    #                     [0.0],      # 'recFTU' as float
    #                     [0.0],      # 'recFTV' as float
    #                     [0.0]       # 'recFTW' as float
    #                 ]
    #             else:
    #                 # Calib data, all zeros, same length as the cluster data
    #                 num_entries = len(event_group["uid"])
    #                 calib_data = [
    #                     [0] * num_entries,  # 'sector' as byte
    #                     [0] * num_entries,  # 'layer' as byte
    #                     [0.0] * num_entries,  # 'size' as float
    #                     [0] * num_entries,  # 'dbstU' as short
    #                     [0] * num_entries,  # 'dbstV' as short
    #                     [0] * num_entries,  # 'dbstW' as short
    #                     [0.0] * num_entries,  # 'rawEU' as float
    #                     [0.0] * num_entries,  # 'rawEV' as float
    #                     [0.0] * num_entries,  # 'rawEW' as float
    #                     [0.0] * num_entries,  # 'recEU' as float
    #                     [0.0] * num_entries,  # 'recEV' as float
    #                     [0.0] * num_entries,  # 'recEW' as float
    #                     [0.0] * num_entries,  # 'recDTU' as float
    #                     [0.0] * num_entries,  # 'recDTV' as float
    #                     [0.0] * num_entries,  # 'recDTW' as float
    #                     [0.0] * num_entries,  # 'recFTU' as float
    #                     [0.0] * num_entries,  # 'recFTV' as float
    #                     [0.0] * num_entries   # 'recFTW' as float
    #                 ]
            
    #             # Write data for the event to the calib bank
    #             file.update({calib_bank: np.array(calib_data)})

    #         # Close the hipo file
    #         file.close()