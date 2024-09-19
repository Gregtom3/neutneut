# Number of model features
N_feat = 30

# Maximum population of strips per event
K = 100

# Train/Test Split Ratio
train_test_ratio = 0.8 # 80%

# Name of model features
model_x_names = [
'energy', 'time', 
'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
'layer_1', 'layer_2', 'layer_3', 'layer_4',
'layer_5', 'layer_6', 'layer_7', 'layer_8', 'layer_9', 
'centroid_x', 'centroid_y', 'centroid_z', 
'is_3way_same_group', 'is_2way_same_group', 'is_3way_cross_group', 'is_2way_cross_group',
'sector_1', 'sector_2', 'sector_3', 
'sector_4','sector_5', 'sector_6']

# Name of model true value
model_y_name = ['unique_otid']

# Name of model misc values
model_misc_names = ['rec_pid', 'pindex', 'mc_pid']

# Scale range for ECAL time
ECAL_time_min = 0
ECAL_time_max = 200

# Scale range for ECAL energy
ECAL_energy_min = 0
ECAL_energy_max = 1

# Scale range for ECAL (x,y) values
ECAL_xy_min = -500
ECAL_xy_max = 500

# Scale range for ECAL (z) values
ECAL_z_min  = 550
ECAL_z_max  = 950

# ECAL_cluster bank names
ECAL_cluster_names = ["id", "status", "sector", "layer", "x", "y", "z", "energy", "time", "widthU", "widthV", "widthW", "idU", "idV", "idW", "coordU", "coordV", "coordW"]

# CSV column names
csv_column_names =   ["event", "id", "mc_pid", "otid", "sector", "layer", "energy", "time", "xo", "yo", "zo", "xe", "ye", "ze", "rec_pid", "pindex"]