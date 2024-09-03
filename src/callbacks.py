import tensorflow as tf
from ModelEcalPlotter import ModelEcalPlotter
import matplotlib.pyplot as plt
from Evaluator import Evaluator
import numpy as np

class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y, misc, tB, tD, outdir=None):
        super().__init__()
        self.X = X
        self.y = y
        self.misc = misc
        self.tB = tB
        self.tD = tD
        self.outdir = outdir

    def on_epoch_end(self, epoch, logs=None):
        
        evaluator = Evaluator(self.X, 
                              self.y, 
                              self.misc)
        evaluator.load_model(self.model)
        evaluator.predict()
        evaluator.cluster(self.tB, self.tD)
        
        for n in range(self.X.shape[0]):
            plotter = ModelEcalPlotter(evaluator.get_event_dataframe(n))
            outfile = f"{self.outdir}/ECAL_{epoch:05}_ev{n}.png" 

            suptitle = f"Epoch {epoch:05}\nLoss = {logs.get('loss'):.4f}"
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
    def __init__(self, save_path):
        super(LossPlotCallback, self).__init__()
        self.save_path = save_path

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

        plt.figure(figsize=(7, 5))

        # Plot all the individual losses on the same plot
        plt.subplot(2, 1, 1)
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
        plt.yscale('log')

        # Plot the total loss on a separate subplot
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_loss, 'k-', label='Train Total Loss')
        plt.plot(epochs, val_loss, 'k--', label='Val Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.yscale('log')
        plt.tight_layout()

        # Save the figure
        plt.savefig(self.save_path)
        plt.close()
        print(f"Training loss plots saved to {self.save_path}")