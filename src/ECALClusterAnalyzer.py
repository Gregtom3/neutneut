import pandas as pd
import numpy as np
import shutil
class ECALClusterAnalyzer:
    """
    Class to analyze ECAL data, create clusters, and return a DataFrame
    containing information about the clusters for each event.
    """

    def __init__(self, input_df, clustering_variable='pindex'):
        """
        Initialize the analyzer with the event DataFrame and define the clustering variable.
        
        Parameters:
        -----------
        input_df : pd.DataFrame
            The DataFrame containing the clustering results
        clustering_variable : str
            The column name used to cluster the data within each event (default is 'pindex').
        """
        self.event_df = input_df
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
            'is_2way_same_group'
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
                status = int(priority_group["cluster_id"].iloc[0])
                widthU = 0
                widthV = 0
                widthW = 0

                idU = 0
                idV = 0
                idW = 0

                coordU = 0
                coordV = 0
                coordW = 0

                return  {
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

    