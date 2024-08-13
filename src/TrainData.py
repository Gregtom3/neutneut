import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

class DataPreprocessor:
    """
    Class to preprocess the data, including filtering, rescaling, column removal, rotation, and one-hot encoding operations.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, df):
        """
        Preprocess the DataFrame by calling individual preprocessing subroutines.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to preprocess.

        Returns:
        --------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        
        df = self._filter_peak_time(df)
        df = self._rotate_coordinates(df)
        df = self._one_hot_encode(df, 'sector', 6)
        df = self._one_hot_encode(df, 'layer_A', 9)
        df = self._one_hot_encode(df, 'layer_B', 9)
        df = self._rescale_columns(df, ['energy_A', 'energy_B', 'time_A', 'time_B','x_rot','y_rot'])
        df = self._delete_columns(df, ['pindex', 'mc_index', 'event_number'])
        df = self._reorder_columns(df)
        
        return df

    def _filter_peak_time(self, df):
        """
        Filter rows where peak time is outside the range 0-200.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to filter.

        Returns:
        --------
        pd.DataFrame
            The filtered DataFrame.
        """
        return df[(df['time_A'] >= 0) & (df['time_A'] <= 200) & (df['time_B'] >= 0) & (df['time_B'] >= 0)].copy()

    def _rescale_columns(self, df, columns_to_scale):
        """
        Rescale specified numeric columns between 0-1.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to rescale.
        columns_to_scale : list of str
            List of column names to scale.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with scaled columns.
        """
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        return df

    def _delete_columns(self, df, columns_to_delete):
        """
        Delete specified columns from the DataFrame if they exist.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to modify.
        columns_to_delete : list of str
            List of column names to delete if they exist in the DataFrame.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with specified columns removed.
        """
        return df.drop(columns=[col for col in columns_to_delete if col in df.columns])
    
    
    def _rotate_coordinates(self, df):
        """
        Rotate the 'x' and 'y' coordinates by (sector-1)*60 degrees clockwise.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame with 'x' and 'y' coordinates to rotate.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with rotated coordinates.
        """
        # Function to rotate a point by theta degrees
        def rotate_point(x, y, theta):
            rad = np.deg2rad(theta)
            cos_theta = np.cos(rad)
            sin_theta = np.sin(rad)
            x_rot = x * cos_theta + y * sin_theta
            y_rot = -x * sin_theta + y * cos_theta
            return x_rot, y_rot

        theta = (df['sector'] - 1) * 60
        df[f'x_rot'], df[f'y_rot'] = rotate_point(df[f'x'], df[f'y'], theta)

        return df    
    
    def _one_hot_encode(self, df, column_name, num_categories):
        """
        One-hot encode a specified column.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to modify.
        column_name : str
            The name of the column to one-hot encode.
        num_categories : int
            The number of unique categories in the column.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with one-hot encoded columns.
        """
        for i in range(1, num_categories + 1):
            df[f'{column_name}_{i}'] = (df[column_name] == i).astype(int)
        df = df.drop(columns=[column_name])
        return df    


    def _reorder_columns(self, df):
        """
        Reorder the DataFrame columns

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to reorder.

        Returns:
        --------
        pd.DataFrame
            The reordered DataFrame.
        """
        # Define the new order of columns
        column_order = [
            'file_number', 'file_event_number', 'global_event_number', 'unique_mc_index', 'mc_pid'
        ] + [col for col in df.columns if col not in [
            'file_number', 'file_event_number', 'global_event_number', 'unique_mc_index', 'mc_pid'
        ]]

        # Reorder the columns
        df = df[column_order]

        return df
    
    
    
class TrainData:
    """
    Class to handle and merge multiple intersection CSV files into a single DataFrame,
    ensuring unique mc_index across all files and splitting the data into train and test sets.
    """

    def __init__(self, csv_files, train_size=0.8, graph=False, k_nearest_neighbors=5, min_intersections=2):
        """
        Initialize the TrainData class with a list of CSV files and the desired train/test split.

        Parameters:
        -----------
        csv_files : list[str]
            List of paths to the intersection CSV files.
        train_size : float
            The proportion of the dataset to include in the train split (default is 0.8).
        graph : bool
            If True, generate graphs instead of dataframes.
        k_nearest_neighbors : int
            Number of nearest neighbors to construct graph.
        min_intersections : int
            Minimum number of intersections required for an event to be included (must be > 1).
        """
        if min_intersections <= 1:
            raise ValueError("min_intersections must be greater than 1")

        self.csv_files = csv_files
        self.train_size = train_size
        self.graph = graph
        self.k_nearest_neighbors = k_nearest_neighbors
        self.min_intersections = min_intersections
        self.data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

        # Automatically load, merge, preprocess, and split the data upon initialization
        self._load_and_merge_csvs()
        self._preprocess_data()
        self._split_data()

    def _load_and_merge_csvs(self):
        """
        Load and merge multiple intersection CSV files into a single DataFrame,
        ensuring unique mc_index and properly tagging event numbers.
        Filter out events with fewer intersections than min_intersections.
        Also, report statistics after loading.
        """
        unique_mc_index_offset = 0  # Offset to ensure unique_mc_index across all files
        global_event_number = 0  # Global event number across all files
        total_events_added = 0  # Total number of events added

        merged_data = []

        for file_number, csv_file in tqdm(enumerate(self.csv_files)):
            # Load the CSV file
            try:
                df = pd.read_csv(csv_file)
            except: # No columns in csv file
                continue
            # Add file_number and global_event_number to the DataFrame
            df['file_number'] = file_number
            df['file_event_number'] = df['event_number']
            df['global_event_number'] = df['event_number'] + global_event_number

            # Filter events based on min_intersections
            intersection_counts = df['global_event_number'].value_counts()
            valid_events = intersection_counts[intersection_counts >= self.min_intersections].index
            df = df[df['global_event_number'].isin(valid_events)]

            # Update the offset for the next file
            if not df.empty:
                if unique_mc_index_offset > 0:
                    df.loc[df['unique_mc_index'] != -1, 'unique_mc_index'] += unique_mc_index_offset
                unique_mc_index_offset = df['unique_mc_index'].max() + 1

            # Update global_event_number for the next file
            global_event_number += df['event_number'].max() + 1

            # Gather statistics
            events_added = len(df)
            total_events_added += events_added

            merged_data.append(df)

        # Concatenate all DataFrames
        self.data = pd.concat(merged_data, ignore_index=True)

        # Check if the merged DataFrame is empty
        if self.data.empty:
            raise ValueError("No data available after loading and merging CSV files. Please check your input files or filtering criteria.")

        # Report statistics
        total_files = len(self.csv_files)
        print(f"Total files processed: {total_files}")
        print(f"Total intersections found: {total_events_added}")

    def _preprocess_data(self):
        """
        Preprocess the merged data using the DataPreprocessor class.
        """
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess(self.data)

    def _split_data(self):
        """
        Split the merged and preprocessed data into train and test sets based on the specified train_size.
        """
        # Shuffle the data
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the data into train and test sets
        train_size = int(len(self.data) * self.train_size)
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]

        if self.graph:
            self.train_data = self._convert_to_graphs(self.train_data)
            self.test_data = self._convert_to_graphs(self.test_data)

    def _convert_to_graphs(self, df):
        """
        Convert a DataFrame into a list of graphs, connecting nodes by nearest neighbors in (x, y) space.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to convert.

        Returns:
        --------
        list of torch_geometric.data.Data
            A list of graphs corresponding to the events in the DataFrame.
        """
        graphs = []
        feature_columns = [
            'energy_A', 'energy_B', 'time_A', 'time_B',
            'x_rot', 'y_rot', 'sector_1', 'sector_2', 'sector_3', 'sector_4',
            'sector_5', 'sector_6', 'layer_A_1', 'layer_A_2', 'layer_A_3',
            'layer_A_4', 'layer_A_5', 'layer_A_6', 'layer_A_7', 'layer_A_8',
            'layer_A_9', 'layer_B_1', 'layer_B_2', 'layer_B_3', 'layer_B_4',
            'layer_B_5', 'layer_B_6', 'layer_B_7', 'layer_B_8', 'layer_B_9'
        ]

        for event_id, group in df.groupby('global_event_number'):
            x = torch.tensor(group[feature_columns].values, dtype=torch.float)
            y = torch.tensor(group['unique_mc_index'].values, dtype=torch.long)

            # Extract (x, y) coordinates for KDTree
            coords = group[['x', 'y']].values
            tree = KDTree(coords)

            # Ensure k doesn't exceed the number of available points
            k = min(self.k_nearest_neighbors + 1, len(coords))

            # Find the nearest neighbors for each point
            edge_index = []
            for i, coord in enumerate(coords):
                distances, neighbors = tree.query(coord, k=k)  # Adjust k based on the available points

                # Ensure neighbors is treated as a 1D array
                neighbors = np.atleast_1d(neighbors)

                for neighbor in neighbors[1:]:  # Exclude the point itself
                    if neighbor < len(coords):
                        edge_index.append([i, neighbor])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            graph = Data(x=x, edge_index=edge_index, y=y)
            graphs.append(graph)

        return graphs

    def get_train_data(self):
        """
        Get the train DataFrame or list of graphs.

        Returns:
        --------
        pd.DataFrame or list of torch_geometric.data.Data
            The DataFrame or list of graphs containing the training data.
        """
        return self.train_data

    def get_test_data(self):
        """
        Get the test DataFrame or list of graphs.

        Returns:
        --------
        pd.DataFrame or list of torch_geometric.data.Data
            The DataFrame or list of graphs containing the testing data.
        """
        return self.test_data

    def save_train_test_to_csv(self, train_filename, test_filename):
        """
        Save the train and test DataFrames to separate CSV files.

        Parameters:
        -----------
        train_filename : str
            The path to the output train CSV file.
        test_filename : str
            The path to the output test CSV file.
        """
        if not self.graph:
            self.train_data.to_csv(train_filename, index=False)
            self.test_data.to_csv(test_filename, index=False)