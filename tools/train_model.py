import os
import sys
import shutil
import yaml
import tensorflow as tf
sys.path.append("/work/clas12/users/gmat/clas12/neutneut/src/")
from TrainData import load_zip_train_test_data
from model_functions import make_gravnet_model
from loss_functions import CustomLoss, AttractiveLossMetric, RepulsiveLossMetric, CowardLossMetric, NoiseLossMetric
from ModelEcalPlotter import ModelEcalPlotter
from Evaluator import Evaluator
from callbacks import PlotCallback, CyclicalLearningRate, LossPlotCallback, PrintBatchMetricsCallback
from helper_functions import create_output_directory, create_gif_from_pngs, check_gpus
import math
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint


def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def count_batches(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

def main(config_path):
    # Check GPUs 
    check_gpus()
    
    # Load the configuration
    config = load_config(config_path)
    
    project_directory = config['project_directory']
    output_dir        = config['output_dir']
    
    # Set training parameters from config
    batch_size = int(config['batch_size'])
    initial_lr = float(config['initial_lr'])

    max_lr = float(config['max_lr'])
    step_size = int(config['step_size'])
    N_epochs = int(config['N_epochs'])
    q_min = float(config['q_min'])
    tB = float(config['tB'])
    tD = float(config['tD'])
    ev = int(config['ev'])
    ev_n = int(config['ev_n'])
    # Get "train_size" and "test_size" from config if available
    train_size = config.get('train_size', None)
    test_size = config.get('test_size', None)

    # Calculate num_train_batches and num_test_batches if train_size and test_size are found
    if train_size is not None:
        num_train_batches = int(math.ceil(train_size / batch_size))  # Round up
    else:
        num_train_batches = None  # Set to None if not found

    if test_size is not None:
        num_test_batches = int(math.ceil(test_size / batch_size))  # Round up
    else:
        num_test_batches = None  # Set to None if not found

    
    # Load training and testing data in batches, but limit to `num_batches`
    (train_X_data, train_y_data, train_misc_data), (test_X_data, test_y_data, test_misc_data) = load_zip_train_test_data(
        project_directory, batch_size, num_train_batches=num_train_batches, num_test_batches = num_test_batches
    )

    # Get the number of training batches now if its all loaded
    if num_train_batches == None:
        num_train_batches = count_batches(train_y_data)

    # Zip the training and testing datasets so that they can be used in model.fit()
    train_data = tf.data.Dataset.zip((train_X_data, train_y_data))
    test_data = tf.data.Dataset.zip((test_X_data, test_y_data))
    
    # Use the provided output directory
    outdir = create_output_directory(output_dir)

    # Create checkpoints directory
    os.makedirs(f'{outdir}/checkpoints',exist_ok=True)

    # Copy the config file to the output directory
    shutil.copy(config_path, os.path.join(outdir, "config.yaml"))

    # Define the model
    model = make_gravnet_model(
        K=100,
        N_feat=28,
        N_grav_layers=config['N_grav_layers'],
        N_neighbors=config['N_neighbors'],
        N_filters=config['N_filters'],
        use_sector=False
    )

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    # Instantiate the custom loss function
    custom_loss = CustomLoss(q_min=q_min)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=[
            AttractiveLossMetric(q_min=q_min),
            RepulsiveLossMetric(q_min=q_min),
            CowardLossMetric(q_min=q_min),
            NoiseLossMetric(q_min=q_min)
        ]
    )

    # Callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config['patience'], restore_best_weights=True
    )

    clr_callback = CyclicalLearningRate(
        initial_learning_rate=initial_lr, 
        max_learning_rate=max_lr, 
        step_size=step_size
    )

    for x in train_X_data.take(1):
        ecal_x_train = x[0:0+ev_n]
        break
    for y in train_y_data.take(1):
        ecal_y_train = y[0:0+ev_n]
        break
    for misc in train_misc_data.take(1):
        ecal_misc_train= misc[0:0+ev_n]
        break
    
    for x in test_X_data.take(1):
        ecal_x_test = x[0:0+ev_n]
        break
    for y in test_y_data.take(1):
        ecal_y_test = y[0:0+ev_n]
        break
    for misc in test_misc_data.take(1):
        ecal_misc_test = misc[0:0+ev_n]
        break


    ecal_train_plot_callback = PlotCallback(
        X=ecal_x_train,
        y=ecal_y_train,
        misc=ecal_misc_train,
        tB=tB,
        tD=tD,
        outdir=outdir,
        version="train"
    )
    
    
    ecal_test_plot_callback = PlotCallback(
        X=ecal_x_test,
        y=ecal_y_test,
        misc=ecal_misc_test,
        tB=tB,
        tD=tD,
        outdir=outdir,
        version="test"
    )

    loss_plot_callback = LossPlotCallback(
        save_dir=outdir+"/../"
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=outdir+'/checkpoints/epoch_{epoch:04d}.keras',  # Save model to this file path, epoch will be part of the filename
        monitor='loss',                                       # You can monitor 'val_loss' or another metric if needed
        save_best_only=False,                                 # Set to True to save only the best model
        save_weights_only=False,                              # Set to True if you only want to save model weights
        save_freq='epoch',                                    # Save after every epoch
        verbose=0                                             # Print a message when the model is saved
    )

    # Output the number of parameters in the model
    print("Number of Parameters:", model.count_params())

    # Train the model
    history = model.fit(train_data,
                        batch_size=batch_size,
                        epochs=N_epochs,
                        shuffle=True,
                        validation_data=test_data,
                        callbacks=[early_stopping_callback,
                                   clr_callback,
                                   ecal_train_plot_callback,
                                   ecal_test_plot_callback,
                                   loss_plot_callback,
                                   PrintBatchMetricsCallback(num_train_batches=num_train_batches,
                                                             num_epochs=N_epochs),
                                   checkpoint_callback
                        ],
                        verbose=2)

    create_gif_from_pngs(outdir, "training.fast", duration=60)
    create_gif_from_pngs(outdir, "training.slow", duration=500)

    # Save the model to the pre-made directory "outdir"
    # Define paths for the two model formats
    model_save_paths = {
        'keras': f"{outdir}/trained_model.keras",   # Native Keras format (.keras)
        'savedmodel': f"{outdir}/saved_model"       # SavedModel directory format
    }

    # Remove existing files or directories if they exist
    for path in model_save_paths.values():
        if os.path.exists(path):
            if os.path.isfile(path):  # Check if it's a file
                os.remove(path)
            else:  # For SavedModel, it's a directory
                shutil.rmtree(path)

    # Save the model in the native .keras format (recommended for new models)
    model.save(model_save_paths['keras'], save_format='keras')

    # Save the model in SavedModel directory format (also recommended)
    model.save(model_save_paths['savedmodel'], save_format='tf')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
