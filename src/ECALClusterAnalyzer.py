import pandas as pd
import numpy as np
import hipopy as hp  
import shutil
class ECALClusterAnalyzer:
    """
    Class to analyze ECAL data, create clusters, and return a DataFrame
    containing information about the clusters for each event.
    """

    def __init__(self, input_df, original_hipofile, clustering_variable='pindex'):
        """
        Initialize the analyzer with the event DataFrame and define the clustering variable.
        
        Parameters:
        -----------
        input_df : pd.DataFrame
            The DataFrame containing the clustering results
        original_hipofile : str
            The path to the original hipo file where the cluster data will be written.
        clustering_variable : str
            The column name used to cluster the data within each event (default is 'pindex').
        """
        self.event_df = input_df
        self.original_hipofile = original_hipofile
        self.new_hipofile = self.original_hipofile.replace(".hipo","_OC.hipo")
        # Copy new_hipofile
        shutil.copy2(self.original_hipofile, self.new_hipofile)
        self.clustering_variable = clustering_variable
        self.clusters_df = pd.DataFrame({
            'event': [0],
            'status': [0],
            'sector': [0],
            'layer': [0],
            'energy': [0],
            'time': [0],
            'x': [0],
            'y': [0],
            'z': [0],
            'widthU': [0],
            'widthV': [0],
            'widthW': [0],
            'idU': [0],
            'idV': [0],
            'idW': [0],
            'coordU': [0],
            'coordV': [0],
            'coordW': [0]
        }).iloc[0:0]
        
    def get_strip2point_distance(self, strip_df, point):
        xe = strip_df['xe']
        ye = strip_df['ye']
        ze = strip_df['ze']
        strip_end_point = np.array([xe, ye, ze])
        dist = np.linalg.norm(point - strip_end_point)
        return dist

    def calculate_strip_energy(self, strip_df, centroid):
        iGain = 1.0
        iADC_to_MEV = 1.0 / 10000.0
        iAttenA = 1.0
        iAttenB = 376.0
        iAttenC = 0.0
        iAttenD = 0.0
        iAttenE = 400.0

        edist = self.get_strip2point_distance(strip_df, centroid)
        EcorrFactor = iAttenA * (np.exp(-edist / iAttenB) + iAttenD * np.exp(-edist / iAttenE)) + iAttenC
        strip_energy = strip_df['energy']
        return strip_energy / EcorrFactor

    def calculate_cluster_energy(self, group_df, centroid):
        Etot = 0.0
        for idx, strip in group_df.iterrows():
            Etot += self.calculate_strip_energy(strip, centroid)
        return Etot

    def calculate_cluster_time(self, group_df, centroid):
        veff = 16.0
        max_energy_index = group_df['energy'].idxmax()
        max_energy_strip = group_df.loc[max_energy_index]
        tdist = self.get_strip2point_distance(max_energy_strip, centroid)
        rawtime = max_energy_strip['time']
        return rawtime - tdist / veff

    def create_cluster_from_layergroup(self, layergroup_df):
        """
        Search for rows that have a '1' in the highest priority column and return the cluster properties.
        """
        priority_columns = [
            'is_3way_same_group',
            'is_2way_same_group',
            'is_3way_cross_group',
            'is_2way_cross_group'
        ]

        # Iterate through each priority column in the given order
        for priority in priority_columns:
            priority_group = layergroup_df[layergroup_df[priority] == 1]
            if not priority_group.empty:
                avg_centroid_x = priority_group['centroid_x'].mean()
                avg_centroid_y = priority_group['centroid_y'].mean()
                avg_centroid_z = priority_group['centroid_z'].mean()

                sector = int(priority_group["sector"].iloc[0])
                layer  = int(((priority_group["layer"].iloc[0]-1) // 3)*3+1)  # Integer division to group layers
                uid = priority_group[self.clustering_variable].iloc[0]  # Clustering variable

                # Calculate additional parameters
                centroid = np.array([avg_centroid_x, avg_centroid_y, avg_centroid_z])
                energy = self.calculate_cluster_energy(priority_group, centroid)
                time = self.calculate_cluster_time(priority_group, centroid)
    
                # Default vars
                status = 0
                widthU = 0
                widthV = 0
                widthW = 0

                idU = 0
                idV = 0
                idW = 0

                coordU = 0
                coordV = 0
                coordW = 0

                return {
                    'uid': uid,
                    'status': status,
                    'layer': layer,
                    'sector': sector,
                    'energy': energy,
                    'time': time,
                    'x': avg_centroid_x,
                    'y': avg_centroid_y,
                    'z': avg_centroid_z,
                    'widthU': widthU,
                    'widthV': widthV,
                    'widthW': widthW,
                    'idU': idU,
                    'idV': idV,
                    'idW': idW,
                    'coordU': coordU,
                    'coordV': coordV,
                    'coordW': coordW
                }

        # If no matching priority rows are found, return None
        return None

    def create_cluster_from_group(self, group_df, event):
        """
        Create clusters for each layer group within a group of strips with the same clustering variable.
        Append the resulting cluster information to the clusters DataFrame.
        """
        layergroups = {
            'group1': [1, 2, 3],
            'group2': [4, 5, 6],
            'group3': [7, 8, 9]
        }

        for group_name, layers in layergroups.items():
            layergroup_df = group_df[group_df['layer'].isin(layers)]
            cluster = self.create_cluster_from_layergroup(layergroup_df)
            
            if cluster:
                cluster['event'] = event
                # Convert cluster to a DataFrame for concatenation
                cluster_df = pd.DataFrame([cluster])
                # Concatenate the new cluster_df to self.clusters_df
                self.clusters_df = pd.concat([self.clusters_df, cluster_df], ignore_index=True)

    def create_clusters_from_event(self, event_df):
        """
        Process a single event and create clusters for each group defined by the clustering variable.
        Only the entries belonging to the dominant sector (if one exists) are used for clustering.
        """
        groups = event_df.groupby(self.clustering_variable)

        for cluster_var_value, group in groups:
            if cluster_var_value == -1:
                continue

            # Count the occurrences of each sector in the group
            sector_counts = group['sector'].value_counts()

            # Identify the dominant sector (the sector with the highest count)
            dominant_sector = sector_counts.idxmax()
            dominant_count = sector_counts.max()

            # Check if the dominant sector is unique (i.e., no tie)
            if (sector_counts == dominant_count).sum() > 1:
                # Skip this group if there is a tie between sectors
                continue

            # Filter the group to include only rows from the dominant sector
            dominant_sector_group = group[group['sector'] == dominant_sector]

            # Extract the event number (assuming it's the same for the entire group)
            event = group['event'].iloc[0]

            # Create clusters from the dominant sector group
            self.create_cluster_from_group(dominant_sector_group, event)
        
    def create_clusters(self):
        """
        Group the event data by 'event' and process each event one by one.
        Returns the full DataFrame with all clusters.
        """
        events = self.event_df.groupby('event')

        for event, event_group in events:
            self.create_clusters_from_event(event_group)
        
        return self.clusters_df

    def write_hipo_file(self):
        """
        Write the cluster data into a new hipo file using the generated clusters DataFrame.
        """
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
            
    def run(self):
        self.create_clusters()
        self.write_hipo_file()