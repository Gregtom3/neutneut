import pandas as pd
import numpy as np
from itertools import combinations
from ECALDataReader import ECALDataReader
import csv
from tqdm import tqdm
from global_params import *
from geometry import calculate_strip_centroid_z, closest_point_between_lines, line_intersection

class ECALDataAnalyzer:
    """
    Class to analyze ECAL data and write the results to a CSV file.
    """

    def __init__(self, input_filename, output_filename="output.csv", data_type="mc", centroid_group_mandate=None):
        """
        Initialize the analyzer with input and output file names.
        
        Parameters:
        -----------
        input_filename : str
            Path to the input hipo file.
        output_filename : str
            Path to the initial output CSV file for hits (default is "output.csv").
        data_type : str
            Type of data to process, either "mc" or "rec". Determines how certain fields are handled.
        centroid_group_mandate : str or None
            Field to mandate matching strips by ("pindex", "otid", or None).
        """
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.data_type = data_type
        self.centroid_group_mandate = centroid_group_mandate
        assert(self.data_type in ["mc", "rec"])
        assert(self.centroid_group_mandate in [None, "pindex", "otid"])


    def read_ecal_data_from_event(self):
        """
        Read and process ECAL data from an event file and write results to a CSV.
        """
        reader = ECALDataReader(self.input_filename)
        
        # Open the CSV file for writing
        with open(self.output_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row
            header = csv_column_names # from global_params.py
            
            csv_writer.writerow(header)

            # Loop over the events and process the data
            for event_number, event in tqdm(enumerate(reader.file)):
                if self.data_type == "mc":
                    ECAL_hits = reader.get_dict("ECAL::hits+")
                else:  # "rec" mode
                    ECAL_hits = reader.get_dict("ECAL::hits")

                if len(ECAL_hits) == 0:  # Skip events without ECAL hits
                    continue

                REC_cal = reader.get_dict("REC::Calorimeter")
                REC_parts = reader.get_dict("REC::Particle")

                # Apply the rec_pid and pindex determination
                def get_rec_pid_and_pindex(row):
                    if self.data_type == "rec" or row['clusterId'] == -1:
                        return pd.Series({'rec_pid': -1, 'pindex': -1})
                    else:
                        pindex = REC_cal[REC_cal["index"] == row['clusterId'] - 1]["pindex"].values[0]
                        rec_pid = REC_parts["pid"].values[pindex]
                        return pd.Series({'rec_pid': rec_pid, 'pindex': pindex})

                # Add rec_pid and pindex columns to ECAL_hits DataFrame
                ECAL_hits[['rec_pid', 'pindex']] = ECAL_hits.apply(get_rec_pid_and_pindex, axis=1)

                # In "rec" mode, set 'otid' and 'pid' to -1
                if self.data_type == "rec":
                    ECAL_hits['otid'] = -1
                    ECAL_hits['pid'] = -1

                # Write the data for each hit to the CSV file
                for _, hit in ECAL_hits.iterrows():
                    row = [
                        event_number, hit['id'], hit['pid'], hit['otid'], hit['sector'], hit['layer'], hit['energy'], hit['time'],
                        hit['xo'], hit['yo'], hit['zo'], hit['xe'], hit['ye'], hit['ze'], hit['rec_pid'], hit['pindex']
                    ]
                    csv_writer.writerow(row)
                    

    def process_hipo(self):
        """
        Process the hipo file to generate a CSV with hits data.
        """
        # Generate the initial hits CSV
        self.read_ecal_data_from_event()
        
        # Add the centroid column
        self.add_centroid_columns(R=6)
        print(f"Processing complete.")

    
    def add_centroid_columns(self, R=6):
        # Load the hits data from the previously generated CSV
        df = pd.read_csv(self.output_filename)

        # Initialize centroid columns with default values
        df['centroid_x'] = 0.0
        df['centroid_y'] = 0.0
        df['centroid_z'] = 0.0
        
        # Initialize one-hot encoded columns for intersection types
        df['is_3way_same_group'] = 0
        df['is_2way_same_group'] = 0
        df['is_3way_cross_group'] = 0
        df['is_2way_cross_group'] = 0

        # Process each event_number and sector group
        df = self.process_groups(df, R)

        # Process cross-layer group intersections for remaining strips
        df = self.process_cross_groups(df, R)

        # Save the updated DataFrame back to the original CSV file
        df.to_csv(self.output_filename, index=False)
        print(f"Centroid columns and intersection types added, CSV saved: {self.output_filename}")

        
    def process_groups(self, df, R):
        """
        Process each group of event_number and sector in the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to process.
        R : float
            The maximum allowed distance for considering a 3-way intersection as valid.

        Returns:
        --------
        pd.DataFrame
            The updated DataFrame with centroids and intersection flags set.
        """
        # Group the data by event number and sector
        grouped = df.groupby(['event', 'sector'])

        # Iterate through each event and sector group
        for (event, sector), group in tqdm(grouped):
            for n_start in [1, 4, 7]:
                layer_group = group[group['layer'].isin([n_start, n_start + 1, n_start + 2])]
                df = self.process_layer_group(df, layer_group, event, sector, R)

        return df

    def process_layer_group(self, df, layer_group, event, sector, R):
        """
        Process each strip within a layer group, searching for 3-way and 2-way intersections.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame being updated.
        layer_group : pd.DataFrame
            The subset of data for a specific event, sector, and layer group.
        event : int
            The event number for the group being processed.
        sector : int
            The sector number for the group being processed.
        R : float
            The maximum allowed distance for considering a 3-way intersection as valid.

        Returns:
        --------
        pd.DataFrame
            The updated DataFrame.
        """
        for _, strip_x in layer_group.iterrows():
            if strip_x['centroid_x'] == 0 and strip_x['centroid_y'] == 0:
                # Search for the most energetic 3-way intersection
                max_energy, best_centroid, matched_strips = self.find_best_3way_intersection(strip_x, layer_group, R)
                if max_energy > 0:
                    df = self.update_centroid(df, event, matched_strips, best_centroid, 'is_3way_same_group')
                    continue

                # If no 3-way found, search for the most energetic 2-way intersection within the same layer group
                max_energy, best_centroid, matched_strips = self.find_best_2way_intersection(strip_x, layer_group)
                if max_energy > 0:
                    df = self.update_centroid(df, event, matched_strips, best_centroid, 'is_2way_same_group')

        return df

    def process_cross_groups(self, df, R):
        """
        Process strips that haven't found a centroid within their layer group, searching across groups.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame being updated.
        R : float
            The maximum allowed distance for considering a 3-way intersection as valid.

        Returns:
        --------
        pd.DataFrame
            The updated DataFrame.
        """
        # Group the data by event number and sector
        grouped = df.groupby(['event', 'sector'])

        # Iterate through each event and sector group
        for (event, sector), group in tqdm(grouped):
            # Separate the layer groups within the current event and sector
            layer_groups = {n_start: group[group['layer'].isin([n_start, n_start + 1, n_start + 2])]
                            for n_start in [1, 4, 7]}

            for n_start, layer_group in layer_groups.items():
                for _, strip_x in layer_group.iterrows():
                    if strip_x['centroid_x'] == 0 and strip_x['centroid_y'] == 0:
                        # Search for 3-way intersections across non-matching layer groups
                        max_energy, best_centroid, matched_strips = self.find_best_3way_intersection(strip_x, group, R, cross_group=True)
                        if max_energy > 0:
                            df = self.update_centroid(df, strip_x['event'], matched_strips, best_centroid, 'is_3way_cross_group')
                            continue

                        # If no 3-way found, search for 2-way intersections across all layer groups
                        max_energy, best_centroid, matched_strips = self.find_best_2way_intersection(strip_x, group, cross_group=True)
                        if max_energy > 0:
                            df = self.update_centroid(df, strip_x['event'], matched_strips, best_centroid, 'is_2way_cross_group')

        return df
    

    def find_best_3way_intersection(self, strip_x, layer_group, R, cross_group=False):
        """
        Find the most energetic 3-way intersection for a given strip.

        Parameters:
        -----------
        strip_x : pd.Series
            The strip being processed.
        layer_group : pd.DataFrame
            The subset of data within the layer group or cross groups.
        R : float
            The maximum allowed distance for considering a 3-way intersection as valid.
        cross_group : bool
            If True, search across all layers, not just within the layer group.

        Returns:
        --------
        tuple : (max_energy, best_centroid, matched_strips)
            max_energy : float
                The energy of the best 3-way intersection found.
            best_centroid : np.array
                The centroid coordinates of the best intersection.
            matched_strips : list
                The IDs of strips involved in the best intersection.
        """
        max_energy = 0
        best_centroid = np.array([0.0, 0.0])
        matched_strips = []

        for _, strip_n1 in layer_group.iterrows():
            for _, strip_n2 in layer_group.iterrows():
                if strip_n1['id'] == strip_n2['id'] or strip_x['id'] == strip_n1['id'] or strip_x['id'] == strip_n2['id']:
                    continue

                # Calculate closest points between lines
                closest_n_n1 = closest_point_between_lines(
                    np.array([strip_x['xo'], strip_x['yo']]),
                    np.array([strip_x['xe'], strip_x['ye']]) - np.array([strip_x['xo'], strip_x['yo']]),
                    np.array([strip_n1['xo'], strip_n1['yo']]),
                    np.array([strip_n1['xe'], strip_n1['ye']]) - np.array([strip_n1['xo'], strip_n1['yo']])
                )
                closest_n_n2 = closest_point_between_lines(
                    np.array([strip_x['xo'], strip_x['yo']]),
                    np.array([strip_x['xe'], strip_x['ye']]) - np.array([strip_x['xo'], strip_x['yo']]),
                    np.array([strip_n2['xo'], strip_n2['yo']]),
                    np.array([strip_n2['xe'], strip_n2['ye']]) - np.array([strip_n2['xo'], strip_n2['yo']])
                )
                closest_n1_n2 = closest_point_between_lines(
                    np.array([strip_n1['xo'], strip_n1['yo']]),
                    np.array([strip_n1['xe'], strip_n1['ye']]) - np.array([strip_n1['xo'], strip_n1['yo']]),
                    np.array([strip_n2['xo'], strip_n2['yo']]),
                    np.array([strip_n2['xe'], strip_n2['ye']]) - np.array([strip_n2['xo'], strip_n2['yo']])
                )

                centroid = (closest_n_n1 + closest_n_n2 + closest_n1_n2) / 3
                distances = [
                    np.linalg.norm(closest_n_n1 - centroid),
                    np.linalg.norm(closest_n_n2 - centroid),
                    np.linalg.norm(closest_n1_n2 - centroid)
                ]

                if all(distance <= R for distance in distances):
                    total_energy = strip_x['energy'] + strip_n1['energy'] + strip_n2['energy']
                    if total_energy > max_energy:
                        max_energy = total_energy
                        best_centroid = centroid
                        matched_strips = [strip_x['id'], strip_n1['id'], strip_n2['id']]

        return max_energy, best_centroid, matched_strips

    def find_best_2way_intersection(self, strip_x, layer_group, cross_group=False):
        """
        Find the most energetic 2-way intersection for a given strip.

        Parameters:
        -----------
        strip_x : pd.Series
            The strip being processed.
        layer_group : pd.DataFrame
            The subset of data within the layer group or cross groups.
        cross_group : bool
            If True, search across all layers, not just within the layer group.

        Returns:
        --------
        tuple : (max_energy, best_centroid, matched_strips)
            max_energy : float
                The energy of the best 2-way intersection found.
            best_centroid : np.array
                The centroid coordinates of the best intersection.
            matched_strips : list
                The IDs of strips involved in the best intersection.
        """
        max_energy = 0
        best_centroid = np.array([0.0, 0.0])
        matched_strips = []

        for _, strip_n1 in layer_group.iterrows():
            if strip_x['id'] == strip_n1['id']:
                continue

            # Calculate 2-way intersection
            intersection = line_intersection(
                np.array([strip_x['xo'], strip_x['yo']]),
                np.array([strip_x['xe'], strip_x['ye']]) - np.array([strip_x['xo'], strip_x['yo']]),
                np.array([strip_n1['xo'], strip_n1['yo']]),
                np.array([strip_n1['xe'], strip_n1['ye']]) - np.array([strip_n1['xo'], strip_n1['yo']])
            )

            if intersection is not None:
                total_energy = strip_x['energy'] + strip_n1['energy']
                if total_energy > max_energy:
                    max_energy = total_energy
                    best_centroid = intersection
                    matched_strips = [strip_x['id'], strip_n1['id']]

        return max_energy, best_centroid, matched_strips

    def update_centroid(self, df, event, matched_strips, centroid, intersection_type):
        """
        Update the centroid columns and one-hot encoded intersection flags in the DataFrame,
        only if all strips involved in the intersection match based on the `centroid_group_mandate`.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame being updated.
        event : int
            The event number of the strips being updated.
        matched_strips : list
            The IDs of the strips involved in the intersection.
        centroid : np.array
            The centroid coordinates to set.
        intersection_type : str
            The type of intersection (one of 'is_3way_same_group', 'is_2way_same_group', 
            'is_3way_cross_group', 'is_2way_cross_group').

        Returns:
        --------
        pd.DataFrame
            The updated DataFrame.
        """
        if self.centroid_group_mandate is not None:
            # Get the values for the centroid_group_mandate (pindex or otid) for the matched strips
            mandate_values = df[(df['event'] == event) & (df['id'].isin(matched_strips))][self.centroid_group_mandate].values

            # Check if all values for the group mandate are the same
            if len(set(mandate_values)) != 1:
                # Skip updating if they do not match
                return df

        # Update the centroid and intersection type for the matched strips
        for strip_id in matched_strips:
            strip = df[(df['event'] == event) & (df['id'] == strip_id)]
            idx = strip.index
            df.loc[idx, 'centroid_x'] = centroid[0]
            df.loc[idx, 'centroid_y'] = centroid[1]
            df.loc[idx, 'centroid_z'] = calculate_strip_centroid_z(strip, centroid[0], centroid[1])
            df.loc[idx, intersection_type] = 1

        return df

    
