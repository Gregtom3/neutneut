import tensorflow as tf
from ModelEcalPlotter import ModelEcalPlotter
import matplotlib.pyplot as plt
from Evaluator import Evaluator
import numpy as np
import pandas as pd
class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, misc, tB, tD, outdir=None, version=None):
        super().__init__()
        self.X = X
        self.y = y
        self.misc = misc
        self.tB = tB
        self.tD = tD
        self.outdir = outdir
        self.version_text = "_"+version if version!=None else "" # Addendum for the pngs
        
    def on_epoch_end(self, epoch, logs=None):
        
        evaluator = Evaluator.from_data(self.X, 
                                        self.y, 
                                        self.misc)
        evaluator.load_model(self.model)
        evaluator.predict()
        evaluator.cluster(self.tB, self.tD)
        loss_df = evaluator.get_loss_df()
        for n in range(self.X.shape[0]):
            plotter = ModelEcalPlotter(evaluator.get_event_dataframe(n), use_clas_calo_scale=True)
            outfile = f"{self.outdir}/ECAL_{epoch:05}_ev{n}{self.version_text}.png" 
            ev_loss_df = loss_df.iloc[n]
            suptitle = f"Epoch {epoch:05}\nEpoch Loss = {logs.get('loss'):.4f}"+"\n"+f"Evt Loss={ev_loss_df.att_loss:.4f}+{ev_loss_df.rep_loss:.4f}+{ev_loss_df.cow_loss:.4f}+{ev_loss_df.nse_loss:.4f}={ev_loss_df.tot_loss:.4f}"
            plotter.plot_all(tD=evaluator.tD,out=outfile,suptitle=suptitle)

        print(f"End of epoch {epoch+1}")
        
        
class CyclicalLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, max_learning_rate, step_size):
        super(CyclicalLearningRate, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.step_size = step_size
        self.batch_count = 0

    def on_batch_begin(self, batch, logs=None):
        cycle = np.floor(1 + self.batch_count / (2 * self.step_size))
        x = np.abs(self.batch_count / self.step_size - 2 * cycle + 1)
        lr = self.initial_learning_rate + (self.max_learning_rate - self.initial_learning_rate) * np.maximum(0, (1 - x))
        self.model.optimizer.learning_rate.assign(lr)
        self.batch_count += 1



class LossPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super(LossPlotCallback, self).__init__()
        self.save_dir = save_dir

    def on_train_end(self, logs=None):
        # Extract the metric values
        epochs = range(1, len(self.model.history.history['loss']) + 1)
        train_loss = self.model.history.history['loss']
        val_loss = self.model.history.history['val_loss']
        train_attractive_loss = self.model.history.history['attractive_loss']
        val_attractive_loss = self.model.history.history['val_attractive_loss']
        train_repulsive_loss = self.model.history.history['repulsive_loss']
        val_repulsive_loss = self.model.history.history['val_repulsive_loss']
        train_coward_loss = self.model.history.history['coward_loss']
        val_coward_loss = self.model.history.history['val_coward_loss']
        train_lp_loss = self.model.history.history['Lp_loss']  # New Lp loss
        val_lp_loss = self.model.history.history['val_Lp_loss']  # Validation Lp loss

        plt.figure(figsize=(7, 12), dpi=100)

        # Plot all the individual losses on the same plot
        plt.subplot(3, 1, 1)
        plt.plot(epochs, train_attractive_loss, 'b-', label='Train Attractive Loss')
        plt.plot(epochs, val_attractive_loss, 'b--', label='Val Attractive Loss')
        plt.plot(epochs, train_repulsive_loss, 'g-', label='Train Repulsive Loss')
        plt.plot(epochs, val_repulsive_loss, 'g--', label='Val Repulsive Loss')
        plt.plot(epochs, train_coward_loss, 'r-', label='Train Coward Loss')
        plt.plot(epochs, val_coward_loss, 'r--', label='Val Coward Loss')
        plt.title('Individual Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot the Lp loss on a separate subplot
        plt.subplot(3, 1, 2)
        plt.plot(epochs, train_lp_loss, 'm-', label='Train Lp Loss')
        plt.plot(epochs, val_lp_loss, 'm--', label='Val Lp Loss')
        plt.title('Lp Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Lp Loss')
        plt.legend()

        # Plot the total loss on a separate subplot
        plt.subplot(3, 1, 3)
        plt.plot(epochs, train_loss, 'k-', label='Train Total Loss')
        plt.plot(epochs, val_loss, 'k--', label='Val Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

        # Save the figure
        fig_path = f"{self.save_dir}/loss.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"Training loss plots saved to {fig_path}")
        
        # Create a DataFrame to store the loss values for each epoch
        loss_data = {
            'Epoch': list(epochs),
            'Train Loss': train_loss,
            'Val Loss': val_loss,
            'Train Attractive Loss': train_attractive_loss,
            'Val Attractive Loss': val_attractive_loss,
            'Train Repulsive Loss': train_repulsive_loss,
            'Val Repulsive Loss': val_repulsive_loss,
            'Train Coward Loss': train_coward_loss,
            'Val Coward Loss': val_coward_loss,
            'Train Lp Loss': train_lp_loss,  # New Lp loss
            'Val Lp Loss': val_lp_loss  # Validation Lp loss
        }

        df_loss = pd.DataFrame(loss_data)

        # Save the DataFrame to CSV
        csv_path = f"{self.save_dir}/loss.csv"
        df_loss.to_csv(csv_path, index=False)
        print(f"Training loss CSV saved to {csv_path}")
        
class PrintBatchMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_train_batches=None, num_epochs=None):
        super().__init__()
        self.num_train_batches = num_train_batches
        self.num_epochs = num_epochs
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1  # Store current epoch, epochs start at 0 in Keras
    
    def on_batch_end(self, batch, logs=None):
        # Only print every 1/100th of the total number of batches
        if self.num_train_batches is not None and batch % max(1, self.num_train_batches // 100) == 0:
            # Get the metrics from the logs dictionary and print them
            loss = logs.get('loss')
            attractive_loss = logs.get('attractive_loss')
            repulsive_loss = logs.get('repulsive_loss')
            coward_loss = logs.get('coward_loss')
            noise_loss = logs.get('noise_loss')
            lp_loss = logs.get('Lp_loss')  # New Lp loss

            # Print metrics with epoch and batch information, including total batches
            print(f"Epoch {self.current_epoch}/{self.num_epochs}, Batch {batch+1}/{self.num_train_batches} - "
                  f"Loss: {loss:.4f}, Attractive: {attractive_loss:.4f}, "
                  f"Repulsive: {repulsive_loss:.4f}, Coward: {coward_loss:.4f}, "
                  f"Noise: {noise_loss:.4f}, Lp: {lp_loss:.4f}", flush=True)

            