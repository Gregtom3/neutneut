import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
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
        df = self._one_hot_encode(df, 'peak_sector', 6)
        df = self._one_hot_encode(df, 'peak_layer', 9)
        df = self._rescale_columns(df, ['peak_energy', 'peak_time', 'peak_xo', 'peak_yo', 'peak_zo', 'peak_xe', 'peak_ye', 'peak_ze'])
        df = self._delete_columns(df, ['pindex', 'mc_index', 'event_number', 'status', 'id'])
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
        return df[(df['peak_time'] >= 0) & (df['peak_time'] <= 200) & (df['peak_time'] >= 0) & (df['peak_time'] >= 0)].copy()

    def _rescale_columns(self, df, columns_to_scale):
        """
        Rescale specified numeric columns between 0-1, with certain groups sharing the same min-max scaling.

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
        # Group 1: Shared scaling for 'peak_xo', 'peak_yo', 'peak_xe', 'peak_ye'
        group_1 = ['peak_xo', 'peak_yo', 'peak_xe', 'peak_ye']
        group_1_values = df[group_1].values.flatten()  # Flatten to find global min and max
        min_val = group_1_values.min()
        max_val = group_1_values.max()
        df[group_1] = (df[group_1] - min_val) / (max_val - min_val)

        # Group 2: Shared scaling for 'peak_zo' and 'peak_ze'
        group_2 = ['peak_zo', 'peak_ze']
        group_2_values = df[group_2].values.flatten()  # Flatten to find global min and max
        min_val = group_2_values.min()
        max_val = group_2_values.max()
        df[group_2] = (df[group_2] - min_val) / (max_val - min_val)

        # Other columns to scale individually
        other_columns = [col for col in columns_to_scale if col not in group_1 + group_2]
        df[other_columns] = self.scaler.fit_transform(df[other_columns])

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
        Reorder the DataFrame columns, and rename

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
            'file_number', 'file_event_number', 'unique_mc_index', 'mc_pid'
        ] + [col for col in df.columns if col not in [
            'file_number', 'file_event_number', 'unique_mc_index', 'mc_pid'
        ]]

        # Reorder the columns
        df = df[column_order]
        
        # Rename the columns by removing the substring 'peak_'
        df.columns = df.columns.str.replace('peak_', '', regex=False)
        
        return df
    
    
    
class TrainData:
    """
    Class to handle and merge multiple intersection CSV files into a single DataFrame,
    ensuring unique mc_index across all files and splitting the data into train and test sets.
    """

    def __init__(self, csv_files, train_size=0.8, return_tensor=False, K=10):
        """
        Initialize the TrainData class with a list of CSV files and the desired train/test split.

        Parameters:
        -----------
        csv_files : list[str]
            List of paths to the intersection CSV files.
        train_size : float
            The proportion of the dataset to include in the train split (default is 0.8).
        return_tensor : bool
            If True, returns the data as tensors instead of DataFrames.
        K : int
            The number of elements to include in each tensor along the second dimension.
        """
        self.csv_files = csv_files
        self.train_size = train_size
        self.return_tensor = return_tensor
        self.K = K
        self.data = pd.DataFrame()
        self.train_data = None
        self.test_data = None

        # Automatically load, merge, preprocess, and split the data upon initialization
        self._load_and_merge_csvs()
        self._preprocess_data()
        self._split_data()

    def _load_and_merge_csvs(self):
        unique_mc_index_offset = 0
        total_events_added = 0
        merged_data = []

        for file_number, csv_file in tqdm(enumerate(self.csv_files), total=len(self.csv_files)):
            try:
                df = pd.read_csv(csv_file)
            except:
                continue

            df['file_number'] = file_number
            df['file_event_number'] = df['event_number']

            if not df.empty:
                if unique_mc_index_offset > 0:
                    df.loc[df['unique_mc_index'] != -1, 'unique_mc_index'] += unique_mc_index_offset
                unique_mc_index_offset = df['unique_mc_index'].max() + 1

            events_added = len(df)
            merged_data.append(df)

        self.data = pd.concat(merged_data, ignore_index=True)

        if self.data.empty:
            raise ValueError("No data available after loading and merging CSV files. Please check your input files or filtering criteria.")

        print(f"Total files processed: {len(self.csv_files)}")

    def _preprocess_data(self):
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess(self.data)

    def _split_data(self):
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(len(self.data) * self.train_size)
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]

        if self.return_tensor:
            self.train_data = self._convert_to_tensor(self.train_data)
            self.test_data = self._convert_to_tensor(self.test_data)

    def _convert_to_tensor(self, df):
        feature_columns = [
            'energy', 'time', 'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
            'sector_1', 'sector_2', 'sector_3', 'sector_4',
            'sector_5', 'sector_6', 'layer_1', 'layer_2', 'layer_3', 'layer_4',
            'layer_5', 'layer_6', 'layer_7', 'layer_8', 'layer_9', 
            'rec_pid', 'pindex', 'mc_pid', 'unique_mc_index'
        ]

        tensors = []
        grouped = df.groupby(['file_number', 'file_event_number'])

        for _, group in grouped:
            group = group.sort_values(by='energy', ascending=False)
            tensor = group[feature_columns].values[:self.K]
            
            if len(tensor) < self.K:
                padding = np.zeros((self.K - len(tensor), len(feature_columns)))
                tensor = np.vstack([tensor, padding])
            
            tensors.append(tensor)

        return tf.convert_to_tensor(tensors, dtype=tf.float32)

    def get_train_data(self):
        return self._get_data(self.train_data)

    def get_test_data(self):
        return self._get_data(self.test_data)
                                                     
    def _get_data(self, data):
        X = data[:, :, :23]
        y = data[:, :, -1:]
        misc = data[:, :, 23:-1]

        # Cast y to int32
        y = tf.cast(y, tf.int32)

        return X, y, misc
                                                     